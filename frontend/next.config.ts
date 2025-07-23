import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  api: {
    bodyParser: {
      sizeLimit: '10mb', // file upload size limit
    },
  },
};

export default nextConfig;
