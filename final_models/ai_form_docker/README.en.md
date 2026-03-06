# AI Form Coach (Short)

- What it does: webcam detects lateral raise, draws skeleton, scores, and shows tips.
- Stack: Vite + React front end (`apps/web`); algorithms in `packages/core`; model placeholder in `packages/models`.
- Run: Node 18+; after `npm install` run `npm run dev -w web`, open http://localhost:5173/ and allow the camera.
- Use: face the camera, perform one raise-lower rep; score and advice appear; right panel shows model status.
- Folders: `apps/web` front end; `packages/core` algorithms; `config` params; `assets/templates` templates; `tools/*` helper scripts.
- Note: template and denoiser are placeholders; angle conventions not fully unified; results are demo only.
