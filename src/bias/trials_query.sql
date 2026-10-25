-- One row per trial with the raw columns used to build the slices.
-- Countries and therapeutic areas can have several values per trial,
-- so they are joined into one text value separated by "|".
SELECT
    s.nct_id,
    s.overall_status,
    s.phase,
    s.study_type,
    s.start_date,
    sp.agency_class AS sponsor_class,
    e.gender,
    e.child,
    e.adult,
    e.older_adult,
    c.countries,
    a.therapeutic_areas
FROM ctgov.studies s
LEFT JOIN ctgov.sponsors sp
    ON sp.nct_id = s.nct_id AND sp.lead_or_collaborator = 'lead'
LEFT JOIN ctgov.eligibilities e
    ON e.nct_id = s.nct_id
LEFT JOIN (
    SELECT nct_id, string_agg(DISTINCT name, '|') AS countries
    FROM ctgov.countries
    GROUP BY nct_id
) c ON c.nct_id = s.nct_id
LEFT JOIN (
    SELECT nct_id, string_agg(DISTINCT downcase_mesh_term, '|')
        AS therapeutic_areas
    FROM ctgov.browse_conditions
    WHERE downcase_mesh_term = ANY(%(areas)s)
    GROUP BY nct_id
) a ON a.nct_id = s.nct_id
