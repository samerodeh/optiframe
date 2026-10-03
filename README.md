# OptiFrame

Mobile-first web app for measuring recycled eyeglass lenses and generating a printable frame.

This first milestone implements the capture experience:

- the user selects whether the lens is loose or still in a frame;
- the user selects the left or right eye;
- the app captures a camera photo or imports an original image;
- the user confirms that the reference card and target are visible;
- FastAPI validates the image and returns a capture receipt without storing it.

Uploads are limited to 4 MB so they remain below Vercel Functions' 4.5 MB request-body limit.

Contour detection, metric measurement, model training, SVG export, and STL generation are intentionally reserved for the next milestones.

## Run locally

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8765
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite forwards `/api` requests to FastAPI on port 8765.

## Vercel deployment

The repository root contains a `vercel.json` with two services:

- `frontend`: Vite, publicly mounted at `/`;
- `backend`: FastAPI, publicly mounted at `/api/*`.

The browser calls `/api/captures` on the same origin, so no production environment variable or internal service binding is required. Run all services locally with:

```powershell
vercel dev
```

Import the repository root into Vercel using the Services preset. The `/api/(.*)` rewrite must remain before the frontend catch-all rewrite.

## API

- `GET /api/health`
- `POST /api/captures`

The capture endpoint accepts multipart form data with `image`, `capture_mode`, and `eye`.
