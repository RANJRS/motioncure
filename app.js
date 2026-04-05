// ==========================================
// MOTIONCURE — App JS
// ==========================================

// ── Navbar scroll effect ──
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
});

// ── Code snippets per tab ──
const codeSnippets = {
  install: {
    filename: 'install.py',
    code: `<span class="cm"># ── Install all required dependencies ──</span>
<span class="op">!</span>pip install opencv-python-headless matplotlib numpy scikit-image pandas

<span class="kw">import</span> cv2
<span class="kw">import</span> numpy <span class="kw">as</span> np
<span class="kw">import</span> os, zipfile
<span class="kw">import</span> matplotlib.pyplot <span class="kw">as</span> plt
<span class="kw">from</span> google.colab <span class="kw">import</span> files
<span class="kw">from</span> skimage.metrics <span class="kw">import</span> structural_similarity <span class="kw">as</span> ssim
<span class="kw">import</span> pandas <span class="kw">as</span> pd

<span class="cm"># ── Create working directories ──</span>
os.<span class="fn">makedirs</span>(<span class="str">"dataset"</span>, exist_ok=<span class="cls">True</span>)
os.<span class="fn">makedirs</span>(<span class="str">"processed_frames"</span>, exist_ok=<span class="cls">True</span>)

<span class="cm"># ── Upload your dataset ──</span>
uploaded = files.<span class="fn">upload</span>()
`
  },
  preprocess: {
    filename: 'preprocess.py',
    code: `<span class="cm"># ── Extract / Convert uploaded files ──</span>
<span class="kw">for</span> filename <span class="kw">in</span> uploaded.<span class="fn">keys</span>():
    <span class="kw">if</span> filename.<span class="fn">endswith</span>(<span class="str">".zip"</span>):
        <span class="kw">with</span> zipfile.<span class="cls">ZipFile</span>(filename, <span class="str">'r'</span>) <span class="kw">as</span> z:
            z.<span class="fn">extractall</span>(<span class="str">"dataset"</span>)

    <span class="kw">elif</span> filename.<span class="fn">endswith</span>((<span class="str">".mp4"</span>, <span class="str">".avi"</span>)):
        cap = cv2.<span class="fn">VideoCapture</span>(filename)
        count = <span class="num">0</span>
        <span class="kw">while</span> <span class="cls">True</span>:
            ret, frame = cap.<span class="fn">read</span>()
            <span class="kw">if not</span> ret: <span class="kw">break</span>
            cv2.<span class="fn">imwrite</span>(f<span class="str">"dataset/frame_{count:04d}.png"</span>, frame)
            count += <span class="num">1</span>
        cap.<span class="fn">release</span>()

<span class="cm"># ── 6-step preprocessing pipeline ──</span>
<span class="kw">def</span> <span class="fn">preprocess</span>(img):
    img     = cv2.<span class="fn">resize</span>(img, (<span class="num">384</span>, <span class="num">384</span>))
    gray    = cv2.<span class="fn">cvtColor</span>(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.<span class="fn">fastNlMeansDenoising</span>(gray, <span class="cls">None</span>, <span class="num">10</span>, <span class="num">7</span>, <span class="num">21</span>)
    clahe   = cv2.<span class="fn">createCLAHE</span>(clipLimit=<span class="num">2.5</span>, tileGridSize=(<span class="num">8</span>,<span class="num">8</span>))
    contrast= clahe.<span class="fn">apply</span>(denoise)
    smooth  = cv2.<span class="fn">bilateralFilter</span>(contrast, <span class="num">7</span>, <span class="num">50</span>, <span class="num">50</span>)
    kernel  = np.<span class="fn">array</span>([[<span class="num">0</span>,<span class="num">-1</span>,<span class="num">0</span>],[<span class="num">-1</span>,<span class="num">5</span>,<span class="num">-1</span>],[<span class="num">0</span>,<span class="num">-1</span>,<span class="num">0</span>]])
    sharp   = cv2.<span class="fn">filter2D</span>(smooth, <span class="num">-1</span>, kernel)
    <span class="kw">return</span> sharp

<span class="kw">for</span> file <span class="kw">in</span> <span class="fn">sorted</span>(os.<span class="fn">listdir</span>(<span class="str">"dataset"</span>)):
    img = cv2.<span class="fn">imread</span>(os.path.<span class="fn">join</span>(<span class="str">"dataset"</span>, file))
    <span class="kw">if</span> img <span class="kw">is not</span> <span class="cls">None</span>:
        cv2.<span class="fn">imwrite</span>(f<span class="str">"processed_frames/{file}"</span>, <span class="fn">preprocess</span>(img))

<span class="fn">print</span>(<span class="str">"Preprocessing Done"</span>)
`
  },
  motion: {
    filename: 'motion_detection.py',
    code: `<span class="cm"># ── Hybrid Motion Detection: Optical Flow + SSIM ──</span>
files_list    = <span class="fn">sorted</span>(os.<span class="fn">listdir</span>(<span class="str">"processed_frames"</span>))
motion_values = []
ssim_motion   = []

<span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(files_list) - <span class="num">1</span>):
    img1 = cv2.<span class="fn">imread</span>(f<span class="str">"processed_frames/{files_list[i]}"</span>,   <span class="num">0</span>)
    img2 = cv2.<span class="fn">imread</span>(f<span class="str">"processed_frames/{files_list[i+1]}"</span>, <span class="num">0</span>)

    <span class="cm"># Gaussian pre-smoothing for stable flow</span>
    img1_blur = cv2.<span class="fn">GaussianBlur</span>(img1, (<span class="num">5</span>,<span class="num">5</span>), <span class="num">0</span>)
    img2_blur = cv2.<span class="fn">GaussianBlur</span>(img2, (<span class="num">5</span>,<span class="num">5</span>), <span class="num">0</span>)

    <span class="cm"># Dense Farneback Optical Flow</span>
    flow = cv2.<span class="fn">calcOpticalFlowFarneback</span>(
        img1_blur, img2_blur, <span class="cls">None</span>,
        pyr_scale=<span class="num">0.5</span>, levels=<span class="num">3</span>, winsize=<span class="num">15</span>,
        iterations=<span class="num">3</span>, poly_n=<span class="num">5</span>, poly_sigma=<span class="num">1.2</span>, flags=<span class="num">0</span>)

    mag = np.<span class="fn">sqrt</span>(flow[...,<span class="num">0</span>]**<span class="num">2</span> + flow[...,<span class="num">1</span>]**<span class="num">2</span>)

    motion_values.<span class="fn">append</span>(np.<span class="fn">percentile</span>(mag, <span class="num">90</span>))   <span class="cm"># 90th pct magnitude</span>
    ssim_motion.<span class="fn">append</span>(<span class="num">1</span> - <span class="fn">ssim</span>(img1, img2))        <span class="cm"># SSIM dissimilarity</span>

<span class="cm"># Adaptive thresholds</span>
threshold_motion = np.<span class="fn">median</span>(motion_values) + <span class="num">1.2</span>*np.<span class="fn">std</span>(motion_values)
threshold_ssim   = np.<span class="fn">mean</span>(ssim_motion)   + <span class="num">0.8</span>*np.<span class="fn">std</span>(ssim_motion)
`
  },
  correct: {
    filename: 'correction.py',
    code: `<span class="cm"># ── Motion Correction via Flow-Guided Remapping ──</span>
artifact_frames  = []
corrected_frames = []

<span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(files_list) - <span class="num">1</span>):
    img1 = cv2.<span class="fn">imread</span>(f<span class="str">"processed_frames/{files_list[i]}"</span>,   <span class="num">0</span>)
    img2 = cv2.<span class="fn">imread</span>(f<span class="str">"processed_frames/{files_list[i+1]}"</span>, <span class="num">0</span>)

    img1_blur = cv2.<span class="fn">GaussianBlur</span>(img1, (<span class="num">5</span>,<span class="num">5</span>), <span class="num">0</span>)
    img2_blur = cv2.<span class="fn">GaussianBlur</span>(img2, (<span class="num">5</span>,<span class="num">5</span>), <span class="num">0</span>)

    flow = cv2.<span class="fn">calcOpticalFlowFarneback</span>(
        img1_blur, img2_blur, <span class="cls">None</span>, <span class="num">0.5</span>, <span class="num">3</span>, <span class="num">15</span>, <span class="num">3</span>, <span class="num">5</span>, <span class="num">1.2</span>, <span class="num">0</span>)

    mag = np.<span class="fn">sqrt</span>(flow[...,<span class="num">0</span>]**<span class="num">2</span> + flow[...,<span class="num">1</span>]**<span class="num">2</span>)

    <span class="cm"># Dual threshold artifact gate</span>
    <span class="kw">if</span> (np.<span class="fn">percentile</span>(mag, <span class="num">90</span>) > threshold_motion <span class="kw">or</span>
        (<span class="num">1</span> - <span class="fn">ssim</span>(img1, img2)) > threshold_ssim):
        artifact_frames.<span class="fn">append</span>(i)

    <span class="cm"># Build identity remap + flow displacement</span>
    h, w = img1.<span class="fn">shape</span>
    flow_map = np.<span class="fn">zeros_like</span>(flow)
    flow_map[...,<span class="num">0</span>] = np.<span class="fn">arange</span>(w)
    flow_map[...,<span class="num">1</span>] = np.<span class="fn">arange</span>(h)[:,<span class="cls">None</span>]

    corrected = cv2.<span class="fn">remap</span>(img2, flow_map + flow, <span class="cls">None</span>, cv2.INTER_LINEAR)
    corrected_frames.<span class="fn">append</span>(corrected)

<span class="fn">print</span>(<span class="str">"Motion Correction Done"</span>)
`
  },
  accuracy: {
    filename: 'accuracy.py',
    code: `<span class="cm"># ── PSNR helper function ──</span>
<span class="kw">def</span> <span class="fn">psnr</span>(a, b):
    mse = np.<span class="fn">mean</span>((a - b)**<span class="num">2</span>)
    <span class="kw">return</span> <span class="num">100</span> <span class="kw">if</span> mse == <span class="num">0</span> <span class="kw">else</span> <span class="num">20</span>*np.<span class="fn">log10</span>(<span class="num">255</span>/np.<span class="fn">sqrt</span>(mse))

<span class="cm"># ── Per-frame accuracy evaluation ──</span>
improved_count = <span class="num">0</span>
ssim_scores    = []
psnr_scores    = []

<span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(corrected_frames)):
    o   = cv2.<span class="fn">imread</span>(f<span class="str">"processed_frames/{files_list[i+1]}"</span>, <span class="num">0</span>)
    c   = corrected_frames[i]
    s   = <span class="fn">ssim</span>(o, c)
    mse = np.<span class="fn">mean</span>((o - c)**<span class="num">2</span>)
    p   = <span class="num">100</span> <span class="kw">if</span> mse == <span class="num">0</span> <span class="kw">else</span> <span class="num">20</span>*np.<span class="fn">log10</span>(<span class="num">255</span>/np.<span class="fn">sqrt</span>(mse))

    ssim_scores.<span class="fn">append</span>(s)
    psnr_scores.<span class="fn">append</span>(p)

    <span class="cm"># Count frame as improved if both thresholds pass</span>
    <span class="kw">if</span> s > <span class="num">0.75</span> <span class="kw">and</span> p > <span class="num">25</span>:
        improved_count += <span class="num">1</span>

accuracy = (improved_count / <span class="fn">len</span>(corrected_frames)) * <span class="num">100</span>

<span class="fn">print</span>(<span class="str">"Average SSIM:"</span>, np.<span class="fn">mean</span>(ssim_scores))
<span class="fn">print</span>(<span class="str">"Average PSNR:"</span>, np.<span class="fn">mean</span>(psnr_scores))
<span class="fn">print</span>(f<span class="str">"Final Improved Accuracy: {accuracy:.2f}%"</span>)
`
  },
  output: {
    filename: 'output.py',
    code: `<span class="cm"># ── Build annotated comparison video ──</span>
h, w = corrected_frames[<span class="num">0</span>].<span class="fn">shape</span>

<span class="kw">def</span> <span class="fn">to_color</span>(x):
    <span class="kw">return</span> cv2.<span class="fn">cvtColor</span>(x, cv2.COLOR_GRAY2BGR)

out = cv2.<span class="cls">VideoWriter</span>(
    <span class="str">'final.avi'</span>, cv2.VideoWriter_fourcc(*<span class="str">'MJPG'</span>),
    <span class="num">2</span>, (w*<span class="num">2</span>, h), <span class="cls">True</span>)

motion_norm = (np.<span class="fn">array</span>(motion_values) - np.<span class="fn">min</span>(motion_values)) / \
              (np.<span class="fn">max</span>(motion_values) - np.<span class="fn">min</span>(motion_values) + <span class="num">1e-6</span>)

<span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(corrected_frames)):
    o = cv2.<span class="fn">imread</span>(f<span class="str">"processed_frames/{files_list[i+1]}"</span>, <span class="num">0</span>)
    c = corrected_frames[i]

    combined = np.<span class="fn">hstack</span>((<span class="fn">to_color</span>(o), <span class="fn">to_color</span>(c)))
    s   = <span class="fn">ssim</span>(o, c)
    mse = np.<span class="fn">mean</span>((o - c)**<span class="num">2</span>)
    p   = <span class="num">100</span> <span class="kw">if</span> mse == <span class="num">0</span> <span class="kw">else</span> <span class="num">20</span>*np.<span class="fn">log10</span>(<span class="num">255</span>/np.<span class="fn">sqrt</span>(mse))

    cv2.<span class="fn">rectangle</span>(combined, (<span class="num">0</span>,<span class="num">0</span>), (w*<span class="num">2</span>,<span class="num">80</span>), (<span class="num">0</span>,<span class="num">0</span>,<span class="num">0</span>), -<span class="num">1</span>)
    cv2.<span class="fn">putText</span>(combined, <span class="str">"ORIGINAL"</span>,  (<span class="num">20</span>,<span class="num">30</span>),  cv2.FONT_HERSHEY_SIMPLEX, <span class="num">0.8</span>, (<span class="num">255</span>,<span class="num">255</span>,<span class="num">255</span>), <span class="num">2</span>)
    cv2.<span class="fn">putText</span>(combined, <span class="str">"CORRECTED"</span>, (w+<span class="num">20</span>,<span class="num">30</span>), cv2.FONT_HERSHEY_SIMPLEX, <span class="num">0.8</span>, (<span class="num">255</span>,<span class="num">255</span>,<span class="num">255</span>), <span class="num">2</span>)
    cv2.<span class="fn">putText</span>(combined, f<span class="str">"SSIM:{s:.3f}"</span>,  (<span class="num">20</span>,<span class="num">65</span>),  cv2.FONT_HERSHEY_SIMPLEX, <span class="num">0.7</span>, (<span class="num">0</span>,<span class="num">255</span>,<span class="num">0</span>),   <span class="num">2</span>)
    cv2.<span class="fn">putText</span>(combined, f<span class="str">"PSNR:{p:.2f}"</span>, (w+<span class="num">20</span>,<span class="num">65</span>), cv2.FONT_HERSHEY_SIMPLEX, <span class="num">0.7</span>, (<span class="num">0</span>,<span class="num">255</span>,<span class="num">255</span>), <span class="num">2</span>)

    <span class="kw">if</span> i <span class="kw">in</span> artifact_frames:
        cv2.<span class="fn">putText</span>(combined, <span class="str">"ARTIFACT DETECTED"</span>,
                    (<span class="fn">int</span>(w*<span class="num">0.5</span>-<span class="num">170</span>), <span class="num">115</span>), cv2.FONT_HERSHEY_SIMPLEX,
                    <span class="num">0.9</span>, (<span class="num">0</span>,<span class="num">0</span>,<span class="num">255</span>), <span class="num">3</span>)

    <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="num">3</span>):
        out.<span class="fn">write</span>(combined)

out.<span class="fn">release</span>()

<span class="cm"># Transcode to H.264 MP4</span>
<span class="op">!</span>ffmpeg -loglevel quiet -i final.avi -vcodec libx264 -crf <span class="num">17</span> final.mp4
files.<span class="fn">download</span>(<span class="str">'final.mp4'</span>)
<span class="fn">print</span>(<span class="str">"🎉 FINAL PROJECT COMPLETE"</span>)
`
  }
};

// ── Tab switching ──
let activeTab = 'install';

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
  const snippet = codeSnippets[tab];
  document.getElementById('code-filename').textContent = snippet.filename;
  document.getElementById('code-text').innerHTML = snippet.code;
  document.getElementById('copy-btn').innerHTML = '<span id="copy-icon">⧉</span> Copy';
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Init first tab
switchTab('install');

// ── Copy code ──
function copyCode() {
  const raw = document.getElementById('code-text').innerText;
  navigator.clipboard.writeText(raw).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.innerHTML = '✓ Copied!';
    btn.style.color = '#48bb78';
    btn.style.borderColor = 'rgba(72,187,120,0.4)';
    setTimeout(() => {
      btn.innerHTML = '<span id="copy-icon">⧉</span> Copy';
      btn.style.color = '';
      btn.style.borderColor = '';
    }, 2000);
  });
}

// ── Intersection observer for metric bars ──
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('animate');
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.metric-bar').forEach(bar => observer.observe(bar));

// ── Smooth count-up for hero stats ──
function animateCount(el, end, suffix = '') {
  let start = 0;
  const isNum = !isNaN(parseInt(end));
  if (!isNum) return; // skip non-numeric
  const num = parseInt(end);
  const step = Math.ceil(num / 30);
  const timer = setInterval(() => {
    start += step;
    if (start >= num) { start = num; clearInterval(timer); }
    el.textContent = start + suffix;
  }, 40);
}

// Trigger once hero visible
const heroObs = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting) {
    animateCount(document.getElementById('stat2'), 9);
    heroObs.disconnect();
  }
}, { threshold: 0.5 });
heroObs.observe(document.querySelector('.hero'));

// ── Stage cards: stagger on scroll ──
const stageCards = document.querySelectorAll('.stage-card');
const cardObs = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

stageCards.forEach((card, i) => {
  card.style.opacity = '0';
  card.style.transform = 'translateY(24px)';
  card.style.transition = `opacity 0.5s ease ${i * 0.07}s, transform 0.5s ease ${i * 0.07}s, border-color 0.3s, box-shadow 0.3s`;
  cardObs.observe(card);
});

// ── General fade-in for sections ──
const sections = document.querySelectorAll('.section, .preprocess-section');
const secObs = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      secObs.unobserve(entry.target);
    }
  });
}, { threshold: 0.05 });

sections.forEach(s => {
  s.style.opacity = '0';
  s.style.transform = 'translateY(32px)';
  s.style.transition = 'opacity 0.7s ease, transform 0.7s ease';
  secObs.observe(s);
});