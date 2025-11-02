import os
import tempfile
import requests
from http.cookiejar import MozillaCookieJar
from flask import Flask, request, jsonify
from flask_caching import Cache
from youtube_search import YoutubeSearch
import yt_dlp
import concurrent.futures
import functools
import threading
import time

# -------------------------
# Use Temp Directory for All File Operations (Vercel/Koyeb/Netlify compatibility)
# -------------------------
# Determine writable temp directory
temp_dir = os.environ.get('TMPDIR', tempfile.gettempdir())
# Ensure the temp dir exists
os.makedirs(temp_dir, exist_ok=True)

# Paths for cookie storage
cookie_file = os.path.join(temp_dir, 'cookies.txt')
cookies_file = cookie_file

# -------------------------
# Load Cookies and Patch requests.get (unchanged)
# -------------------------
if os.path.exists(cookie_file):
    cookie_jar = MozillaCookieJar(cookie_file)
    cookie_jar.load(ignore_discard=True, ignore_expires=True)
    session = requests.Session()
    session.cookies = cookie_jar
    original_get = requests.get

    def get_with_cookies(url, **kwargs):
        kwargs.setdefault('cookies', session.cookies)
        return original_get(url, **kwargs)

    requests.get = get_with_cookies

# -------------------------
# Flask App Initialization
# -------------------------
app = Flask(__name__)

# -------------------------
# Cache Configuration (In-Memory)
# -------------------------
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',  # In-memory
    'CACHE_DEFAULT_TIMEOUT': 0  # "Infinite" until invalidated
})

# -------------------------
# Tuneable concurrency + sensible defaults
# set YT_CONCURRENT_FRAGMENTS in env to control fragment concurrency
# -------------------------
DEFAULT_CONCURRENT_FRAGMENTS = int(os.environ.get('YT_CONCURRENT_FRAGMENTS', '3'))
# Thread pool for running blocking yt_dlp tasks (keeps Flask worker threads free)
YDLP_THREADPOOL_MAX_WORKERS = int(os.environ.get('YDLP_WORKERS', '4'))
_ytdlp_executor = concurrent.futures.ThreadPoolExecutor(max_workers=YDLP_THREADPOOL_MAX_WORKERS)

# -------------------------
# Helper: Convert durations to ISO 8601
# -------------------------
def to_iso_duration(duration_str: str) -> str:
    parts = duration_str.split(':') if duration_str else []
    iso = 'PT'
    if len(parts) == 3:
        h, m, s = parts
        if int(h): iso += f"{int(h)}H"
        iso += f"{int(m)}M{int(s)}S"
    elif len(parts) == 2:
        m, s = parts
        iso += f"{int(m)}M{int(s)}S"
    elif len(parts) == 1 and parts[0].isdigit():
        iso += f"{int(parts[0])}S"
    else:
        iso += '0S'
    return iso

# -------------------------
# yt-dlp Options and Extraction
# - add: concurrent_fragment_downloads
# - add: paths -> temp to ensure temporary fragments go to temp_dir
# - keep cachedir False for ephemeral hosts
# -------------------------
common_ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'skip_download': True,
    'format': 'bestvideo+bestaudio/best',
    'cookiefile': cookies_file,
    'cachedir': False,
    # make yt-dlp write temp/fragment files to our temp dir (avoid repo dir chaos)
    'paths': {'temp': temp_dir},
    # native concurrent fragment downloader setting (works for HLS/DASH)
    'concurrent_fragment_downloads': DEFAULT_CONCURRENT_FRAGMENTS,
    # do not print progress to keep stdout clean on serverless
    'noprogress': True,
}

ydl_opts_full = dict(common_ydl_opts)
# Keep original meta options but add concurrency & temp path
ydl_opts_meta = dict(common_ydl_opts, simulate=True, noplaylist=True, skip_download=True)

# -------------------------
# Helper: run yt_dlp.extract_info in a threadpool with optional timeout
# -------------------------
def _run_extract_info(ydl_opts, target, download=False):
    """
    Blocking call to extract_info using the provided ydl options.
    This runs inside a worker thread via executor below.
    """
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(target, download=download)

def extract_info(url=None, search_query=None, opts=None, timeout=None):
    """
    Run yt-dlp extract_info on a threadpool worker.
    - opts: yt-dlp options dict
    - timeout: seconds to wait for result; if None, wait indefinitely.
    Returns (info, err, code) similar to your original function.
    """
    ydl_opts = opts or ydl_opts_full
    target = None
    if search_query:
        # for ytsearch: prefix the query as in CLI
        target = f"ytsearch:{search_query}"
    else:
        target = url

    future = _ytdlp_executor.submit(_run_extract_info, ydl_opts, target)
    try:
        info = future.result(timeout=timeout)
        # If ytsearch returned a search result dict, the top-level structure can be search results:
        if search_query and isinstance(info, dict) and 'entries' in info:
            entries = info.get('entries') or []
            if not entries:
                return None, {'error': 'No search results'}, 404
            return entries[0], None, None
        return info, None, None
    except concurrent.futures.TimeoutError:
        # Cancel the running future if it did not finish in time; best-effort
        future.cancel()
        return None, {'error': 'yt-dlp timed out'}, 504
    except yt_dlp.utils.DownloadError as e:
        return None, {'error': str(e)}, 500
    except Exception as e:
        return None, {'error': str(e)}, 500

# -------------------------
# Format Helpers (unchanged)
# -------------------------
def get_size_bytes(fmt):
    return fmt.get('filesize') or fmt.get('filesize_approx') or 0

def format_size(bytes_val):
    if bytes_val >= 1e9: return f"{bytes_val/1e9:.2f} GB"
    if bytes_val >= 1e6: return f"{bytes_val/1e6:.2f} MB"
    if bytes_val >= 1e3: return f"{bytes_val/1e3:.2f} KB"
    return f"{bytes_val} B"

def build_formats_list(info):
    fmts = []
    for f in info.get('formats', []):
        url_f = f.get('url')
        if not url_f: continue
        has_video = f.get('vcodec') != 'none'
        has_audio = f.get('acodec') != 'none'
        kind = 'progressive' if has_video and has_audio else \
               'video-only' if has_video else \
               'audio-only' if has_audio else None
        if not kind: continue
        size = get_size_bytes(f)
        fmts.append({
            'format_id': f.get('format_id'),
            'ext': f.get('ext'),
            'kind': kind,
            'filesize_bytes': size,
            'filesize': format_size(size),
            'width': f.get('width'),
            'height': f.get('height'),
            'fps': f.get('fps'),
            'abr': f.get('abr'),
            'asr': f.get('asr'),
            'url': url_f
        })
    return fmts

# -------------------------
# Flask Routes (mostly unchanged) but using the threaded extract_info
# - metadata endpoints use a short timeout to return fast (configurable)
# -------------------------
@app.route('/')
def home():
    key = 'home'
    if 'latest' in request.args:
        cache.delete(key)
    data = cache.get(key)
    if data:
        return jsonify(data)
    data = {'message': '✅ YouTube API is alive'}
    cache.set(key, data)
    return jsonify(data)

@app.route('/api/fast-meta')
def api_fast_meta():
    q = request.args.get('search', '').strip()
    u = request.args.get('url', '').strip()
    key = f"fast_meta:{q}:{u}"
    if 'latest' in request.args:
        cache.delete(key)
    cached = cache.get(key)
    if cached is not None:
        return jsonify(cached)
    if not q and not u:
        return jsonify({'error': 'Provide either "search" or "url" parameter'}), 400
    result = None
    try:
        if q:
            results = YoutubeSearch(q, max_results=1).to_dict()
            if results:
                vid = results[0]
                result = {
                    'title': vid.get('title'),
                    'link': f"https://www.youtube.com/watch?v={vid.get('url_suffix').split('v=')[-1]}",
                    'duration': to_iso_duration(vid.get('duration', '')),
                    'thumbnail': vid.get('thumbnails', [None])[0]
                }
        else:
            # Use a short timeout for metadata so endpoint returns fast (tunable)
            meta_timeout = int(os.environ.get('META_TIMEOUT', '6'))  # seconds
            info, err, code = extract_info(u, None, opts=ydl_opts_meta, timeout=meta_timeout)
            if err:
                return jsonify(err), code
            result = {
                'title': info.get('title'),
                'link': info.get('webpage_url'),
                'duration': to_iso_duration(str(info.get('duration'))),
                'thumbnail': info.get('thumbnail')
            }
        if not result:
            return jsonify({'error': 'No results'}), 404
        cache.set(key, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/all')
def api_all():
    q = request.args.get('search', '').strip()
    u = request.args.get('url', '').strip()
    if not (q or u):
        return jsonify({'error': 'Provide "url" or "search"'}), 400
    # For full info, allow a longer timeout (or None to wait indefinitely)
    full_timeout = int(os.environ.get('FULL_INFO_TIMEOUT', '30'))  # seconds
    info, err, code = extract_info(u or None, q or None, opts=ydl_opts_full, timeout=full_timeout)
    if err:
        return jsonify(err), code
    fmts = build_formats_list(info)
    suggestions = [
        {'id': rel.get('id'),
         'title': rel.get('title'),
         'url': rel.get('webpage_url') or rel.get('url'),
         'thumbnail': rel.get('thumbnails', [{}])[0].get('url')}
        for rel in info.get('related', [])
    ]
    data = {
        'title': info.get('title'),
        'video_url': info.get('webpage_url'),
        'duration': info.get('duration'),
        'upload_date': info.get('upload_date'),
        'view_count': info.get('view_count'),
        'like_count': info.get('like_count'),
        'thumbnail': info.get('thumbnail'),
        'description': info.get('description'),
        'tags': info.get('tags'),
        'is_live': info.get('is_live'),
        'age_limit': info.get('age_limit'),
        'average_rating': info.get('average_rating'),
        'channel': {
            'name': info.get('uploader'),
            'url': info.get('uploader_url') or info.get('channel_url'),
            'id': info.get('uploader_id')
        },
        'formats': fmts,
        'suggestions': suggestions
    }
    return jsonify(data)

@app.route('/api/meta')
def api_meta():
    q = request.args.get('search', '').strip()
    u = request.args.get('url', '').strip()
    key = f"meta:{q}:{u}"
    if 'latest' in request.args:
        cache.delete(key)
    cached = cache.get(key)
    if cached:
        return jsonify(cached)
    if not (q or u):
        return jsonify({'error': 'Provide "url" or "search"'}), 400
    meta_timeout = int(os.environ.get('META_TIMEOUT', '6'))  # keep metadata quick
    info, err, code = extract_info(u or None, q or None, opts=ydl_opts_meta, timeout=meta_timeout)
    if err:
        return jsonify(err), code
    keys = ['id','title','webpage_url','duration','upload_date',
            'view_count','like_count','thumbnail','description',
            'tags','is_live','age_limit','average_rating',
            'uploader','uploader_url','uploader_id']
    data = {'metadata': {k: info.get(k) for k in keys}}
    cache.set(key, data)
    return jsonify(data)

@app.route('/api/channel')
def api_channel():
    cid = request.args.get('id', '').strip()
    cu = request.args.get('url', '').strip()
    key = f"channel:{cid or cu}"
    if 'latest' in request.args:
        cache.delete(key)
    cached = cache.get(key)
    if cached:
        return jsonify(cached)
    if not (cid or cu):
        return jsonify({'error': 'Provide "url" or "id" parameter for channel'}), 400
    try:
        info, err, code = extract_info(cid or cu, None, opts=ydl_opts_meta, timeout=20)
        if err:
            return jsonify(err), code
        data = {
            'id': info.get('id'),
            'name': info.get('uploader'),
            'url': info.get('webpage_url'),
            'description': info.get('description'),
            'subscriber_count': info.get('subscriber_count'),
            'video_count': info.get('channel_follower_count') or info.get('video_count'),
            'thumbnails': info.get('thumbnails'),
        }
        cache.set(key, data)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlist')
def api_playlist():
    pid = request.args.get('id', '').strip()
    pu = request.args.get('url', '').strip()
    key = f"playlist:{pid or pu}"
    if 'latest' in request.args:
        cache.delete(key)
    cached = cache.get(key)
    if cached:
        return jsonify(cached)
    if not (pid or pu):
        return jsonify({'error': 'Provide "url" or "id" parameter for playlist'}), 400
    try:
        info, err, code = extract_info(pid or pu, None, opts=ydl_opts_full, timeout=60)
        if err:
            return jsonify(err), code
        videos = [{
            'id': e.get('id'),
            'title': e.get('title'),
            'url': e.get('webpage_url'),
            'duration': e.get('duration')
        } for e in info.get('entries', [])]
        data = {
            'id': info.get('id'),
            'title': info.get('title'),
            'url': info.get('webpage_url'),
            'item_count': info.get('playlist_count'),
            'videos': videos
        }
        cache.set(key, data)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Instagram/Twitter/TikTok/Facebook routes same pattern
@app.route('/api/instagram')
def api_instagram():
    u = request.args.get('url', '').strip()
    key = f"instagram:{u}"
    if 'latest' in request.args:
        cache.delete(key)
    cached = cache.get(key)
    if cached:
        return jsonify(cached)
    if not u:
        return jsonify({'error': 'Provide "url" parameter for Instagram'}), 400
    try:
        info, err, code = extract_info(u, None, opts=ydl_opts_meta, timeout=20)
        if err:
            return jsonify(err), code
        cache.set(key, info)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/twitter')
def api_twitter():
    u = request.args.get('url', '').strip()
    key = f"twitter:{u}"
    if 'latest' in request.args:
        cache.delete(key)
    cached = cache.get(key)
    if cached:
        return jsonify(cached)
    if not u:
        return jsonify({'error': 'Provide "url" parameter for Twitter'}), 400
    try:
        info, err, code = extract_info(u, None, opts=ydl_opts_meta, timeout=20)
        if err:
            return jsonify(err), code
        cache.set(key, info)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tiktok')
def api_tiktok():
    u = request.args.get('url', '').strip()
    key = f"tiktok:{u}"
    if 'latest' in request.args:
        cache.delete(key)
    cached = cache.get(key)
    if cached:
        return jsonify(cached)
    if not u:
        return jsonify({'error': 'Provide "url" parameter for TikTok'}), 400
    try:
        info, err, code = extract_info(u, None, opts=ydl_opts_meta, timeout=20)
        if err:
            return jsonify(err), code
        cache.set(key, info)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/facebook')
def api_facebook():
    u = request.args.get('url', '').strip()
    key = f"facebook:{u}"
    if 'latest' in request.args:
        cache.delete(key)
    cached = cache.get(key)
    if cached:
        return jsonify(cached)
    if not u:
        return jsonify({'error': 'Provide "url" parameter for Facebook'}), 400
    try:
        info, err, code = extract_info(u, None, opts=ydl_opts_meta, timeout=20)
        if err:
            return jsonify(err), code
        cache.set(key, info)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------
# Stream Endpoints (no caching)
# -------------------------
STREAM_TIMEOUT = 5 * 3600

@app.route('/download')
@cache.cached(timeout=STREAM_TIMEOUT, key_prefix=lambda: f"download:{request.full_path}")
def api_download():
    url = request.args.get('url')
    search = request.args.get('search')
    if not (url or search):
        return jsonify({'error': 'Provide "url" or "search"'}), 400
    # downloads require the full extract_info so give a generous timeout or none
    info, err, code = extract_info(url, search, opts=ydl_opts_full, timeout=None)
    if err:
        return jsonify(err), code
    return jsonify({'formats': build_formats_list(info)})

@app.route('/api/audio')
def api_audio():
    url = request.args.get('url')
    search = request.args.get('search')
    if not (url or search):
        return jsonify({'error': 'Provide "url" or "search"'}), 400
    info, err, code = extract_info(url, search, opts=ydl_opts_full, timeout=30)
    if err:
        return jsonify(err), code
    afmts = [f for f in build_formats_list(info) if f['kind'] in ('audio-only','progressive')]
    return jsonify({'audio_formats': afmts})

@app.route('/api/video')
def api_video():
    url = request.args.get('url')
    search = request.args.get('search')
    if not (url or search):
        return jsonify({'error': 'Provide "url" or "search"'}), 400
    info, err, code = extract_info(url, search, opts=ydl_opts_full, timeout=30)
    if err:
        return jsonify(err), code
    vfmts = [f for f in build_formats_list(info) if f['kind'] in ('video-only','progressive')]
    return jsonify({'video_formats': vfmts})
