import { NextRequest, NextResponse } from 'next/server';

// Proxy GitLab API requests to avoid CORS issues
export async function GET(request: NextRequest) {
  return handleProxy(request);
}

export async function POST(request: NextRequest) {
  return handleProxy(request);
}

export async function PUT(request: NextRequest) {
  return handleProxy(request);
}

export async function DELETE(request: NextRequest) {
  return handleProxy(request);
}

async function handleProxy(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const target = searchParams.get('target');

  if (!target) {
    return NextResponse.json({ error: 'Missing "target" query param (GitLab host)' }, { status: 400 });
  }

  // Build the target URL: https://<host>/api/v4/<path>?<params>
  const slug = request.nextUrl.pathname.replace('/api/gitlab-proxy', '');
  const forwardParams = new URLSearchParams();
  searchParams.forEach((value, key) => {
    if (key !== 'target') {
      forwardParams.append(key, value);
    }
  });
  const queryString = forwardParams.toString();
  const targetUrl = `https://${target}/api/v4${slug}${queryString ? '?' + queryString : ''}`;

  // Forward headers (including PRIVATE-TOKEN for auth)
  const headers: Record<string, string> = {};
  request.headers.forEach((value, key) => {
    // Skip host and connection headers
    if (!['host', 'connection', 'content-length'].includes(key.toLowerCase())) {
      headers[key] = value;
    }
  });

  try {
    const body = request.method !== 'GET' && request.method !== 'HEAD'
      ? await request.text()
      : undefined;

    const res = await fetch(targetUrl, {
      method: request.method,
      headers: {
        ...headers,
        'Accept': 'application/json',
      },
      body,
    });

    const data = await res.text();
    return new NextResponse(data, {
      status: res.status,
      headers: {
        'Content-Type': res.headers.get('content-type') || 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });
  } catch (err) {
    return NextResponse.json({ error: `Proxy error: ${err}` }, { status: 502 });
  }
}

export function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, PRIVATE-TOKEN',
    },
  });
}
