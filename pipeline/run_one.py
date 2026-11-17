"""Makes and publishes ONE video from a queue file: render -> quality check -> video -> upload.
If the quality check fails, nothing is uploaded.   usage: run_one.py queue/<file>.json"""
import json, os, subprocess, sys, datetime, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))
from recipes import DYNAMIC   # sounds that swell on purpose (waves, gusts) get a wider jump allowance
CHANNELS = {"stormwood": ("UCasC3K1be1EwfBzD7Sg1S6A", "YOUTUBE_REFRESH_TOKEN_STORMWOOD"),
            "valley": ("UC0GYz8SesCGxNZut0_I_BIg", "YOUTUBE_REFRESH_TOKEN_VALLEY")}


def already_uploaded(yt, chan_id, title):
    """True if this channel already has a video (public, private or scheduled) with exactly this title."""
    req = yt.playlistItems().list(part="snippet", playlistId="UU" + chan_id[2:], maxResults=50)
    while req is not None:
        resp = req.execute()
        if any(i["snippet"]["title"] == title for i in resp.get("items", [])):
            return True
        req = yt.playlistItems().list_next(req, resp)
    return False


def sh(*a):
    print("+", " ".join(map(str, a)), flush=True); subprocess.run(list(map(str, a)), check=True)


def main():
    qf = sys.argv[1]; job = json.load(open(qf)); name = os.path.splitext(os.path.basename(qf))[0]
    work = os.environ.get("WORK", "/tmp/work"); os.makedirs(work, exist_ok=True)
    audio, video, thumb = f"{work}/{name}.flac", f"{work}/{name}.mp4", f"{work}/{name}.jpg"
    secs = int(job["hours"] * 3600)
    # 0 sign in, confirm the channel, and prove YouTube accepts the title/description (5 s private test, deleted)
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    chan_id, tok = CHANNELS[job["channel"]]
    creds = Credentials(None, refresh_token=os.environ[tok], token_uri="https://oauth2.googleapis.com/token",
                        client_id=os.environ["YOUTUBE_CLIENT_ID"], client_secret=os.environ["YOUTUBE_CLIENT_SECRET"])
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    mine = yt.channels().list(part="id", mine=True).execute()["items"][0]["id"]
    assert mine == chan_id, f"wrong channel: {mine}"
    if os.path.exists(f"{ROOT}/done/{name}.json") or already_uploaded(yt, chan_id, job["title"]):
        print("already published - skipping (no duplicate upload)"); return
    status = {"selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}
    when = datetime.datetime.fromisoformat(job["publish_at"].replace("Z", "+00:00"))
    if when > datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=20):
        status.update(privacyStatus="private", publishAt=when.strftime("%Y-%m-%dT%H:%M:%SZ"))
    else:
        status["privacyStatus"] = "public"
    body = {"snippet": {"title": job["title"], "description": job["description"], "tags": job.get("tags", []),
                        "categoryId": "10", "defaultLanguage": "en"},
            "status": status}
    test = f"{work}/{name}_preflight.mp4"
    sh("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:d=5", "-f", "lavfi",
       "-i", "anullsrc=r=48000:cl=stereo", "-t", "5", "-c:v", "libx264", "-c:a", "aac", test)
    pb = json.loads(json.dumps(body)); pb["status"].pop("publishAt", None); pb["status"]["privacyStatus"] = "private"
    t = yt.videos().insert(part="snippet,status", body=pb, notifySubscribers=False,
                           media_body=MediaFileUpload(test, resumable=True)).execute()
    yt.videos().delete(id=t["id"]).execute(); print("preflight ok", flush=True)
    # 1 image (kept in the repo once fetched)
    img = os.path.join(ROOT, job["image"])
    if not os.path.exists(img):
        os.makedirs(os.path.dirname(img), exist_ok=True)
        urllib.request.urlretrieve(job["image_url"], img)
    # 2 sound + 3 Audio Quality Engineer (exit code 1 stops everything here)
    sh(sys.executable, f"{ROOT}/engine/render_long.py", job["recipe"], secs, job["seed"], audio, job.get("lufs", -17))
    sh(sys.executable, f"{ROOT}/engine/quality_check_long.py", audio, job.get("lufs", -17), secs, 6 if job["recipe"] in DYNAMIC else 3)
    # 4 video (still scene, 1080p) + thumbnail
    sh("ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "1", "-i", img, "-i", audio,
       "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,format=yuv420p",
       "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-crf", "20", "-g", "10",
       "-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-t", secs, "-movflags", "+faststart", video)
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(ROOT, job.get("thumb", job["image"])), "-vf",
       "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720", "-q:v", "3", thumb)
    # 5 accuracy check: real length of the finished video must match the title's promise
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", video]))
    assert abs(dur - secs) < 5, f"video is {dur}s, expected {secs}s"
    assert f"{job['hours']:g} Hour" in job["title"], "title length does not match the video"
    assert len(job["title"]) <= 100
    # 6 upload (checked again: another run may have uploaded it while this one was rendering)
    if already_uploaded(yt, chan_id, job["title"]):
        print("already published by another run - skipping (no duplicate upload)"); return
    req = yt.videos().insert(part="snippet,status", body=body, notifySubscribers=True,
                             media_body=MediaFileUpload(video, chunksize=64 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        st, resp = req.next_chunk(num_retries=8)
        if st: print(f"uploaded {int(st.progress() * 100)}%", flush=True)
    vid = resp["id"]; print("VIDEO", vid, status)
    try:
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(thumb)).execute()
    except Exception as e:
        print("thumbnail not set:", e)
    os.makedirs(f"{ROOT}/done", exist_ok=True)
    json.dump({"video_id": vid, "url": f"https://youtu.be/{vid}", "status": status, "title": job["title"],
               "uploaded": datetime.datetime.now(datetime.timezone.utc).isoformat()},
              open(f"{ROOT}/done/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
