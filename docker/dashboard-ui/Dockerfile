FROM node:20-alpine AS builder

WORKDIR /app

# Install deps
COPY dashboard/ui/package.json dashboard/ui/package-lock.json* ./
RUN npm install

# Copy source and build
COPY dashboard/ui/ .
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
