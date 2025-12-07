
SQL_MAIN = """
SELECT
  b.bus_id,
  s.date_id::date AS date,
  b.region_types,
  d.season,
  s.trip_distance_km, s.avg_speed_kmh, s.passengers_avg,
  s.temperature_avg_c, s.engine_hours_total, s.mileage_total_km,
  COALESCE(s.failure_flag,FALSE)     AS failure_flag,
  COALESCE(s.maintenance_flag,FALSE) AS maintenance_flag,
  f.failure_type,
  COALESCE(p_bridge.part_name, p_kw.part_name) AS part_name,
  COALESCE(p_bridge.expected_lifetime_km, p_kw.expected_lifetime_km) AS expected_lifetime_km
FROM public.fact_bus_status_star s
JOIN public.dim_bus_star b ON b.bus_sk = s.bus_sk
JOIN public.dim_date d      ON d.date_id = s.date_id
LEFT JOIN public.dim_fault f ON f.fault_id = s.fault_id
LEFT JOIN public.bridge_fault_part bfp ON bfp.fault_id = f.fault_id
LEFT JOIN public.dim_part p_bridge     ON p_bridge.part_id = bfp.part_id
LEFT JOIN LATERAL (
  SELECT p2.part_name, p2.expected_lifetime_km
  FROM public.dim_part p2
  WHERE
    (f.failure_type ILIKE '%Brake%'       AND p2.part_name ILIKE '%Brake%') OR
    (f.failure_type ILIKE '%Engine%'      AND (p2.part_name ILIKE '%Engine%' OR p2.part_name ILIKE '%Oil%' OR p2.part_name ILIKE '%Filter%')) OR
    (f.failure_type ILIKE '%Electrical%'  AND p2.part_name ILIKE '%Elect%') OR
    (f.failure_type ILIKE '%Cooling%'     AND (p2.part_name ILIKE '%Radiator%' OR p2.part_name ILIKE '%Cool%')) OR
    (f.failure_type ILIKE '%Transmission%'AND p2.part_name ILIKE '%Trans%') OR
    (f.failure_type ILIKE '%Suspension%'  AND p2.part_name ILIKE '%Suspens%')
  ORDER BY p2.expected_lifetime_km NULLS LAST
  LIMIT 1
) p_kw ON TRUE
"""
