import { BILLING_SERVICE, proxyFetch } from "@/lib/api";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const token = req.cookies.get("dana_token")?.value;
  if (!token) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }
  const window = req.nextUrl.searchParams.get("window") || "daily";
  // In production, extract tenant_id from token. For now use a placeholder.
  const tenantId = "default";
  const res = await proxyFetch(
    BILLING_SERVICE,
    `/v1/usage/${tenantId}?window=${window}`
  );
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
