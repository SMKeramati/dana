"use client";

import { useState, useEffect, useCallback } from "react";

interface ApiKeyData {
  id: number;
  name: string;
  prefix: string;
  key?: string;
  permissions: string[];
  created_at: string;
  last_used: string | null;
}

export function useApiKeys() {
  const [keys, setKeys] = useState<ApiKeyData[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchKeys = useCallback(async () => {
    try {
      const res = await fetch("/api/keys");
      if (res.ok) {
        const data = await res.json();
        setKeys(data);
      }
    } catch {
      // Silently fail - will show empty state
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  async function createKey(name: string, permissions: string[] = ["chat"]): Promise<ApiKeyData | null> {
    const res = await fetch("/api/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, permissions }),
    });
    if (res.ok) {
      const newKey = await res.json();
      setKeys((prev) => [newKey, ...prev]);
      return newKey;
    }
    return null;
  }

  async function deleteKey(id: number): Promise<boolean> {
    const res = await fetch(`/api/keys/${id}`, { method: "DELETE" });
    if (res.ok || res.status === 204) {
      setKeys((prev) => prev.filter((k) => k.id !== id));
      return true;
    }
    return false;
  }

  return { keys, loading, createKey, deleteKey, refresh: fetchKeys };
}
