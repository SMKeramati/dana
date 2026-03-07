/**
 * Backend service URLs (internal, not exposed to browser).
 * In production these resolve via K8s service discovery.
 */
const AUTH_SERVICE = process.env.AUTH_SERVICE_URL || "http://localhost:8001";
const BILLING_SERVICE = process.env.BILLING_SERVICE_URL || "http://localhost:8003";
const API_GATEWAY = process.env.API_GATEWAY_URL || "http://localhost:8000";

export { AUTH_SERVICE, BILLING_SERVICE, API_GATEWAY };

/**
 * Helper to proxy requests to backend services from Next.js API routes.
 */
export async function proxyFetch(
  serviceUrl: string,
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const url = `${serviceUrl}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });
  return res;
}
