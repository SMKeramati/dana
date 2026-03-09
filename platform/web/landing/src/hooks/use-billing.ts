"use client";

import { useState, useEffect, useCallback } from "react";

interface Plan {
  id: string;
  name: string;
  daily_token_limit: number;
  rpm_limit: number;
  price_monthly: number;
}

interface BillingData {
  plans: Plan[];
}

export function useBilling() {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchPlans = useCallback(async () => {
    try {
      const res = await fetch("/api/billing/plans");
      if (res.ok) {
        const data = await res.json();
        setPlans(Array.isArray(data) ? data : Object.values(data));
      }
    } catch {
      // Silently fail
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPlans();
  }, [fetchPlans]);

  return { plans, loading, refresh: fetchPlans };
}
