# Sequence Video & Audio Tool — Cloud Deployment

## Deploy to Railway (free, ~5 minutes)

### Step 1 — Push to GitHub
1. Create a free account at github.com if you don't have one
2. Create a new repository (call it `sequence-video-tool`)
3. Upload all these files to it (drag and drop works on GitHub)

### Step 2 — Deploy on Railway
1. Go to railway.app and sign up with your GitHub account
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your `sequence-video-tool` repository
4. Railway will detect the config and deploy automatically
5. Click **"Generate Domain"** to get your public URL

That's it. Your tool will be live at something like:
`https://sequence-video-tool-production.up.railway.app`

Share that URL with your colleagues — they just open it in any browser.

---

## Making updates

When you change `app.py` or `index.html`:
1. Upload the updated file to GitHub (or use `git push` if you're comfortable with that)
2. Railway redeploys automatically — colleagues get the update instantly

---

## Free tier limits (Railway)

- 500 hours/month compute (more than enough for a team tool)
- Files are stored temporarily during rendering and cleaned up automatically
- If you hit limits, the $5/month Hobby plan is plenty

---

## How colleagues use it

1. Open the URL in Chrome or Safari
2. Click **"Click to select your audio folder"** → navigate to their Google Drive folder → select all audio files
3. Click **"Click to browse"** for the visual → select their image
4. Click **"Start rendering"**
5. Download links appear when done — files download straight to their Downloads folder

---

## Notes

- The tool processes files on Railway's servers — files are automatically deleted after download
- Rendering a typical video (image + 10 audio tracks) takes 2-5 minutes on Railway's free tier
- Sessions are isolated — colleagues can use the tool at the same time without interfering
