FROM node:20-alpine AS builder

WORKDIR /app

# Install deps
COPY dashboard/ui/package.json dashboard/ui/package-lock.json* ./
RUN npm install

# Copy source
COPY dashboard/ui/ .

# Next.js inlines NEXT_PUBLIC_* at BUILD time, not runtime.
# Default to localhost:8001 (docker-compose maps dashboard-api 8000→8001).
ARG NEXT_PUBLIC_API_URL=http://localhost:8001
ARG NEXT_PUBLIC_WS_URL=ws://localhost:8001/ws/live
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL

RUN npm run build

# ── Production stage ──────────────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

ENV NODE_ENV=production
ENV PORT=3001

EXPOSE 3001

ENTRYPOINT ["npx", "next", "start", "-p", "3001"]
