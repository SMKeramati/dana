import { BILLING_SERVICE, proxyFetch } from "@/lib/api";
import { NextResponse } from "next/server";

export async function GET() {
  const res = await proxyFetch(BILLING_SERVICE, "/v1/plans");
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
