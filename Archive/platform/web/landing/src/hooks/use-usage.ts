"use client";

import { useState, useEffect, useCallback } from "react";

interface UsageData {
  total_tokens: number;
  total_requests: number;
  daily: { date: string; tokens: number; requests: number }[];
}

export function useUsage(window: string = "7d") {
  const [data, setData] = useState<UsageData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchUsage = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/usage?window=${window}`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch {
      // Silently fail - will show empty state
    } finally {
      setLoading(false);
    }
  }, [window]);

  useEffect(() => {
    fetchUsage();
  }, [fetchUsage]);

  return { data, loading, refresh: fetchUsage };
}
