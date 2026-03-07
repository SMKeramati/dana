import { AUTH_SERVICE, proxyFetch } from "@/lib/api";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const res = await proxyFetch(AUTH_SERVICE, "/auth/register", {
    method: "POST",
    body: JSON.stringify({ email: body.email, password: body.password }),
  });
  const data = await res.json();
  if (!res.ok) {
    return NextResponse.json(data, { status: res.status });
  }
  // Auto-login after registration
  const loginRes = await proxyFetch(AUTH_SERVICE, "/auth/login", {
    method: "POST",
    body: JSON.stringify({ email: body.email, password: body.password }),
  });
  const loginData = await loginRes.json();
  if (!loginRes.ok) {
    return NextResponse.json(data); // Return user but no auto-login
  }
  const response = NextResponse.json(data);
  response.cookies.set("dana_token", loginData.access_token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: loginData.expires_in || 3600,
    path: "/",
  });
  return response;
}
