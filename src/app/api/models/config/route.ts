import { NextResponse } from 'next/server';

const FIXED_MODEL_CONFIG = {
  providers: [
    {
      id: 'direct',
      name: 'DeepSeek',
      supportsCustomModel: false,
      models: [
        {
          id: 'deepseek-v4-flash',
          name: 'DeepSeek Chat',
        },
      ],
    },
  ],
  defaultProvider: 'direct',
};

export async function GET() {
  return NextResponse.json(FIXED_MODEL_CONFIG);
}

// Handle OPTIONS requests for CORS if needed
export function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}
