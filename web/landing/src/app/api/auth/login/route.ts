import { AUTH_SERVICE, proxyFetch } from "@/lib/api";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const res = await proxyFetch(AUTH_SERVICE, "/auth/login", {
    method: "POST",
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    return NextResponse.json(data, { status: res.status });
  }
  // Set token as httpOnly cookie for security
  const response = NextResponse.json({ email: body.email });
  response.cookies.set("dana_token", data.access_token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: data.expires_in || 3600,
    path: "/",
  });
  return response;
}
