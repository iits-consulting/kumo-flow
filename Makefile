.PHONY: up down test

# backend port; override with `make up PORT=8001`
PORT ?= 8000
# frontend dev port; override with `make up WEB_PORT=5174`
WEB_PORT ?= 5173

# restart-safe: stop any running instance first so `make up` never stacks servers
up: down
	@sleep 1
	cd backend && PORT=$(PORT) uv run main.py > ../backend.log 2>&1 &
	cd frontend && VITE_API_URL=http://localhost:$(PORT) npm run dev -- --port $(WEB_PORT) --strictPort > ../frontend.log 2>&1 &
	@echo "backend http://localhost:$(PORT)  frontend http://localhost:$(WEB_PORT)  (logs: backend.log, frontend.log)"

# ponytail: kill by port, not PID files — also cleans up strays started by hand.
# --strictPort in `up` stops vite silently falling back to 5174+, so these two
# ports are the whole story and this stays correct.
down:
	-fuser -k -TERM $(PORT)/tcp $(WEB_PORT)/tcp

test:
	cd backend && uv run --extra agent pytest -q
