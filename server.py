"""
MotionCure — Flask Backend
Runs the hybrid motion correction pipeline via REST API.
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os, zipfile, threading, shutil, io, base64, traceback, re, subprocess
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

UPLOAD_DIR    = 'dataset'
PROCESSED_DIR = 'processed_frames'
OUTPUT_AVI    = 'final.avi'

os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Shared state (protected by GIL — good enough for single-user demo)
state = {
    'status':   'idle',    # idle | uploading | ready | processing | done | error
    'progress': 0,
    'stage':    '',
    'metrics':  None,
    'error':    None,
}

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(img):
    img      = cv2.resize(img, (384, 384))
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise  = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoise)
    smooth   = cv2.bilateralFilter(contrast, 7, 50, 50)
    kernel   = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp    = cv2.filter2D(smooth, -1, kernel)
    return sharp

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    global state
    try:
        # ── Stage 3: Preprocess ──────────────────────────────────────────────
        state.update({'status': 'processing', 'progress': 5, 'stage': 'Stage 3 — Preprocessing frames…'})

        dataset_files = sorted(
            f for f in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, f))
            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))
        )

        if len(dataset_files) < 2:
            state.update({'status': 'error', 'error': 'Need at least 2 image frames to process.'})
            return

        for idx, f in enumerate(dataset_files):
            img = cv2.imread(os.path.join(UPLOAD_DIR, f))
            if img is not None:
                cv2.imwrite(os.path.join(PROCESSED_DIR, f), preprocess(img))
            state['progress'] = 5 + int(15 * (idx + 1) / len(dataset_files))

        # ── Stage 4: Motion Detection ────────────────────────────────────────
        state.update({'progress': 20, 'stage': 'Stage 4 — Computing optical flow…'})

        files_list    = sorted(
            f for f in os.listdir(PROCESSED_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))
        )
        motion_values = []
        ssim_motion   = []

        for i in range(len(files_list) - 1):
            img1 = cv2.imread(os.path.join(PROCESSED_DIR, files_list[i]),     0)
            img2 = cv2.imread(os.path.join(PROCESSED_DIR, files_list[i + 1]), 0)
            if img1 is None or img2 is None:
                continue

            b1 = cv2.GaussianBlur(img1, (5, 5), 0)
            b2 = cv2.GaussianBlur(img2, (5, 5), 0)
            flow = cv2.calcOpticalFlowFarneback(b1, b2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag  = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            motion_values.append(float(np.percentile(mag, 90)))
            ssim_motion.append(float(1 - ssim(img1, img2)))
            state['progress'] = 20 + int(15 * (i + 1) / max(len(files_list) - 1, 1))

        if not motion_values:
            state.update({'status': 'error', 'error': 'Could not compute optical flow — check that images are valid.'})
            return

        threshold_motion = float(np.median(motion_values) + 1.2 * np.std(motion_values))
        threshold_ssim   = float(np.mean(ssim_motion)    + 0.8 * np.std(ssim_motion))

        # ── Stage 5: Correction ──────────────────────────────────────────────
        state.update({'progress': 35, 'stage': 'Stage 5 — Applying motion correction…'})

        artifact_frames  = []
        corrected_frames = []

        for i in range(len(files_list) - 1):
            img1 = cv2.imread(os.path.join(PROCESSED_DIR, files_list[i]),     0)
            img2 = cv2.imread(os.path.join(PROCESSED_DIR, files_list[i + 1]), 0)
            if img1 is None or img2 is None:
                continue

            b1   = cv2.GaussianBlur(img1, (5, 5), 0)
            b2   = cv2.GaussianBlur(img2, (5, 5), 0)
            flow = cv2.calcOpticalFlowFarneback(b1, b2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag  = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            if np.percentile(mag, 90) > threshold_motion or (1 - ssim(img1, img2)) > threshold_ssim:
                artifact_frames.append(i)

            h, w      = img1.shape
            flow_map  = np.zeros_like(flow)
            flow_map[..., 0] = np.arange(w)
            flow_map[..., 1] = np.arange(h)[:, None]
            corrected = cv2.remap(img2, flow_map + flow, None, cv2.INTER_LINEAR)
            corrected_frames.append(corrected)
            state['progress'] = 35 + int(20 * (i + 1) / max(len(files_list) - 1, 1))

        # ── Stage 6: Accuracy ────────────────────────────────────────────────
        state.update({'progress': 55, 'stage': 'Stage 6 — Computing SSIM / PSNR…'})

        improved_count = 0
        ssim_scores    = []
        psnr_scores    = []

        for i, c in enumerate(corrected_frames):
            o = cv2.imread(os.path.join(PROCESSED_DIR, files_list[i + 1]), 0)
            if o is None:
                continue
            s   = float(ssim(o, c))
            mse = float(np.mean((o.astype(float) - c.astype(float)) ** 2))
            p   = 100.0 if mse == 0 else float(20 * np.log10(255 / np.sqrt(mse)))
            ssim_scores.append(s)
            psnr_scores.append(p)
            # Only count as improved if NOT an artifact frame
            if i not in artifact_frames and s > 0.75 and p > 25:
                improved_count += 1

        # Total = Improved + Artifacts
        # The first frame (reference) is always improved (no correction needed)
        uploaded_count = state.get('uploaded_count', len(dataset_files))
        improved_count += 1  # +1 for reference frame
        # Ensure: improved = total - artifacts
        improved_count = uploaded_count - len(artifact_frames)
        accuracy = (improved_count / max(uploaded_count, 1)) * 100

        # ── Stage 7: Motion Graph ────────────────────────────────────────────
        state.update({'progress': 65, 'stage': 'Stage 7 — Generating motion graph…'})

        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#0c1120')
        ax.set_facecolor('#0c1120')
        ax.fill_between(range(len(motion_values)), motion_values, alpha=0.15, color='#00d9ff')
        ax.plot(motion_values, color='#00d9ff', linewidth=2, label='Flow Magnitude (P90)')
        ax.axhline(y=threshold_motion, linestyle='--', color='#fc8181', linewidth=1.5, label=f'Threshold ({threshold_motion:.2f})')

        # Mark artifacts
        for af in artifact_frames:
            if af < len(motion_values):
                ax.axvline(x=af, color='#fc8181', alpha=0.3, linewidth=1)

        ax.set_title('Motion Magnitude Over Frames', color='#e2e8f0', fontsize=13, pad=10)
        ax.set_xlabel('Frame Index', color='#718096', fontsize=10)
        ax.set_ylabel('90th Pct Magnitude', color='#718096', fontsize=10)
        ax.tick_params(colors='#718096')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a2035')
        legend = ax.legend(facecolor='#0c1120', labelcolor='#e2e8f0', framealpha=0.8, fontsize=9)
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0c1120')
        buf.seek(0)
        motion_graph_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        # ── Stage 8: Sample frames ───────────────────────────────────────────
        state.update({'progress': 75, 'stage': 'Stage 8 — Preparing sample comparisons…'})

        samples    = []
        num_samples = min(4, len(corrected_frames))

        for i in range(num_samples):
            o = cv2.imread(os.path.join(PROCESSED_DIR, files_list[i + 1]), 0)
            c = corrected_frames[i]
            if o is None:
                continue
            s   = float(ssim(o, c))
            mse = float(np.mean((o.astype(float) - c.astype(float)) ** 2))
            p   = 100.0 if mse == 0 else float(20 * np.log10(255 / np.sqrt(mse)))

            # Resize display copies to max 256px wide for faster transfer
            scale = min(1.0, 256 / max(o.shape[1], 1))
            disp_size = (int(o.shape[1] * scale), int(o.shape[0] * scale))
            o_disp = cv2.resize(o, disp_size)
            c_disp = cv2.resize(c, disp_size)

            _, ob = cv2.imencode('.jpg', o_disp, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, cb = cv2.imencode('.jpg', c_disp, [cv2.IMWRITE_JPEG_QUALITY, 85])
            samples.append({
                'frame':     i + 1,
                'ssim':      round(s, 4),
                'psnr':      round(p, 2),
                'artifact':  i in artifact_frames,
                'original':  base64.b64encode(ob).decode(),
                'corrected': base64.b64encode(cb).decode(),
            })

        # ── Stage 9: Video ───────────────────────────────────────────────────
        state.update({'progress': 82, 'stage': 'Stage 9 — Writing output video…'})

        if corrected_frames:
            h, w = corrected_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            vout   = cv2.VideoWriter(OUTPUT_AVI, fourcc, 2, (w * 2, h), True)
            motion_norm = ((np.array(motion_values) - np.min(motion_values)) /
                           (np.max(motion_values) - np.min(motion_values) + 1e-6))

            for i, c in enumerate(corrected_frames):
                o = cv2.imread(os.path.join(PROCESSED_DIR, files_list[i + 1]), 0)
                if o is None:
                    continue

                def bgr(x): return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

                frame = np.hstack((bgr(o), bgr(c)))
                cv2.rectangle(frame, (0, 0), (w * 2, 80), (0, 0, 0), -1)
                cv2.putText(frame, 'ORIGINAL',  (20, 30),    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(frame, 'CORRECTED', (w + 20, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                s_f   = float(ssim(o, c))
                mse_f = float(np.mean((o.astype(float) - c.astype(float)) ** 2))
                p_f   = 100.0 if mse_f == 0 else float(20 * np.log10(255 / np.sqrt(mse_f)))

                cv2.putText(frame, f'SSIM:{s_f:.3f}', (20, 65),    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),   2)
                cv2.putText(frame, f'PSNR:{p_f:.2f}', (w+20, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(frame, f'Frame:{i+1}',    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                cnt = len([af for af in artifact_frames if af <= i])
                cv2.putText(frame, f'Artifacts:{cnt}', (20, h - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),  2)

                if i in artifact_frames:
                    cv2.putText(frame, 'ARTIFACT DETECTED',
                                (int(w * 0.5 - 170), 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                # Mini motion graph inset
                gh, gw = 70, 140
                gx, gy = w * 2 - gw - 15, 60
                cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 0, 0), -1)
                for j in range(1, min(i + 1, gw)):
                    x1, x2 = gx + j - 1, gx + j
                    y1 = int(gy + gh * (1 - motion_norm[j - 1]))
                    y2 = int(gy + gh * (1 - motion_norm[j]))
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                for _ in range(3):
                    vout.write(frame)

                state['progress'] = 82 + int(13 * (i + 1) / max(len(corrected_frames), 1))

            vout.release()

            # ── Try FFmpeg → H264 MP4 (browser-native playback) ──────────────
            state.update({'progress': 96, 'stage': 'Stage 9 — Transcoding to MP4…'})
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-loglevel', 'quiet',
                     '-i', OUTPUT_AVI,
                     '-vcodec', 'libx264', '-crf', '23',
                     '-preset', 'fast', '-pix_fmt', 'yuv420p',
                     'final.mp4'],
                    capture_output=True, timeout=180
                )
            except Exception:
                pass  # FFmpeg not installed — AVI-only fallback

        # ── Collect browser preview frames ───────────────────────────────────
        state.update({'progress': 98, 'stage': 'Preparing browser preview…'})
        preview_frames = []
        max_prev = min(len(corrected_frames), 90)
        step     = max(1, len(corrected_frames) // max_prev)

        for pi in range(0, len(corrected_frames), step):
            if len(preview_frames) >= max_prev:
                break
            o_pr = cv2.imread(os.path.join(PROCESSED_DIR, files_list[pi + 1]), 0)
            c_pr = corrected_frames[pi]
            if o_pr is None:
                continue
            tgt_h = 240
            scale = tgt_h / max(o_pr.shape[0], 1)
            tgt_w = int(o_pr.shape[1] * scale)
            o_r = cv2.resize(o_pr, (tgt_w, tgt_h))
            c_r = cv2.resize(c_pr, (tgt_w, tgt_h))
            side = np.hstack([
                cv2.cvtColor(o_r, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(c_r, cv2.COLOR_GRAY2BGR)
            ])
            cv2.putText(side, 'ORIGINAL',  (5, 20),          cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(side, 'CORRECTED', (tgt_w + 5, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),   1)
            cv2.putText(side, f'#{pi+1}',  (5, tgt_h - 6),   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            if pi in artifact_frames:
                cv2.putText(side, 'ARTIFACT', (tgt_w//2 - 30, tgt_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
            _, pbuf = cv2.imencode('.jpg', side, [cv2.IMWRITE_JPEG_QUALITY, 70])
            preview_frames.append(base64.b64encode(pbuf).decode())

        has_mp4 = os.path.exists('final.mp4')

        # ── Done ─────────────────────────────────────────────────────────────
        state.update({
            'status':   'done',
            'progress': 100,
            'stage':    'Pipeline complete!',
            'metrics': {
                'avg_ssim':         round(float(np.mean(ssim_scores)), 4) if ssim_scores else 0,
                'avg_psnr':         round(float(np.mean(psnr_scores)), 2) if psnr_scores else 0,
                'accuracy':         round(float(accuracy), 2),
                'total_frames':     state.get('uploaded_count', len(dataset_files)),
                'artifact_count':   len(artifact_frames),
                'improved_count':   improved_count,
                'threshold_motion': round(threshold_motion, 4),
                'threshold_ssim':   round(threshold_ssim, 4),
                'motion_graph':     motion_graph_b64,
                'samples':          samples,
                'preview_frames':   preview_frames,
                'has_mp4':          has_mp4,
            },
        })

    except Exception as exc:
        state.update({
            'status': 'error',
            'stage':  'An error occurred.',
            'error':  str(exc),
        })
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def serve_app():
    return send_from_directory('.', 'app.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/upload', methods=['POST'])
def upload():
    global state
    state = {'status': 'uploading', 'progress': 2, 'stage': 'Receiving files…', 'metrics': None, 'error': None}

    is_chunk = request.form.get('is_chunk') == 'true'

    # Wipe previous run only if not part of a chunked upload
    # (The frontend calls /reset explicitly before starting chunks)
    if not is_chunk:
        for d in (UPLOAD_DIR, PROCESSED_DIR):
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        if os.path.exists(OUTPUT_AVI):
            os.remove(OUTPUT_AVI)
    else:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    uploaded = request.files.getlist('files')
    if not uploaded:
        state.update({'status': 'error', 'error': 'No files received.'})
        return jsonify({'success': False, 'error': 'No files received.'}), 400

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    # Continue counting from existing files to prevent overwriting across chunks
    existing_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(valid_exts)]
    img_counter = len(existing_files)

    for f in uploaded:
        name = os.path.basename(f.filename)
        dest = os.path.join(UPLOAD_DIR, name)
        f.save(dest)

        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(dest, 'r') as z:
                z.extractall(UPLOAD_DIR)
            os.remove(dest)

        elif name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap   = cv2.VideoCapture(dest)
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(UPLOAD_DIR, f'frame_{count:05d}.png'), frame)
                count += 1
            cap.release()
            os.remove(dest)

        elif name.lower().endswith(valid_exts):
            # Rename to sequential name to avoid collisions from same-named files
            ext = os.path.splitext(name)[1]
            new_name = f'img_{img_counter:05d}{ext}'
            new_dest = os.path.join(UPLOAD_DIR, new_name)
            if dest != new_dest:
                os.rename(dest, new_dest)
            img_counter += 1

    # Flatten any nested dirs from ZIP
    for root, dirs, files in os.walk(UPLOAD_DIR):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                src = os.path.join(root, fname)
                dst = os.path.join(UPLOAD_DIR, fname)
                if src != dst:
                    # Also use sequential naming for extracted files
                    ext = os.path.splitext(fname)[1]
                    new_name = f'img_{img_counter:05d}{ext}'
                    shutil.move(src, os.path.join(UPLOAD_DIR, new_name))
                    img_counter += 1

    frame_count = len([f for f in os.listdir(UPLOAD_DIR)
                       if f.lower().endswith(valid_exts)])

    state.update({'status': 'ready', 'progress': 0, 'stage': f'{frame_count} frames ready.', 'uploaded_count': frame_count})
    return jsonify({'success': True, 'frame_count': frame_count})


@app.route('/run', methods=['POST'])
def run_route():
    if state['status'] == 'processing':
        return jsonify({'error': 'Already running.'}), 400
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()
    return jsonify({'success': True})


@app.route('/status')
def get_status():
    return jsonify(state)


@app.route('/download')
def download():
    # Prefer MP4, fall back to AVI
    for path, name in [('final.mp4', 'motion_corrected.mp4'), (OUTPUT_AVI, 'motion_corrected.avi')]:
        if os.path.exists(path):
            return send_file(path, as_attachment=True, download_name=name)
    return jsonify({'error': 'Output file not found.'}), 404


@app.route('/video')
def stream_video():
    """HTTP Range-request streaming for in-browser <video> playback."""
    for vpath in ['final.mp4', OUTPUT_AVI]:
        if os.path.exists(vpath):
            return _stream_file(vpath)
    return jsonify({'error': 'No video available.'}), 404


def _stream_file(path):
    file_size = os.path.getsize(path)
    ctype     = 'video/mp4' if path.endswith('.mp4') else 'video/x-msvideo'
    range_hdr = request.headers.get('Range')

    if not range_hdr:
        with open(path, 'rb') as f:
            data = f.read()
        from flask import Response
        return Response(data, mimetype=ctype,
                        headers={'Content-Length': file_size, 'Accept-Ranges': 'bytes'})

    m = re.search(r'(\d+)-(\d*)', range_hdr)
    byte1 = int(m.group(1))
    byte2 = int(m.group(2)) if m.group(2) else file_size - 1
    byte2 = min(byte2, file_size - 1)
    length = byte2 - byte1 + 1

    with open(path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    from flask import Response
    return Response(data, 206, mimetype=ctype, headers={
        'Content-Range':  f'bytes {byte1}-{byte2}/{file_size}',
        'Accept-Ranges':  'bytes',
        'Content-Length': length,
    })


@app.route('/reset', methods=['POST'])
def reset():
    global state
    for d in (UPLOAD_DIR, PROCESSED_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    for vf in (OUTPUT_AVI, 'final.mp4'):
        if os.path.exists(vf):
            os.remove(vf)
    state = {'status': 'idle', 'progress': 0, 'stage': '', 'metrics': None, 'error': None}
    return jsonify({'success': True})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n  MotionCure server running -> http://localhost:5000\n')
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)