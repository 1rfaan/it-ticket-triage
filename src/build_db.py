import duckdb

con = duckdb.connect('data/tickets.duckdb')

con.execute("""
    CREATE OR REPLACE TABLE tickets AS
    SELECT * FROM read_csv_auto('data/processed/tickets_enriched.csv')
""")

print(con.execute("SELECT COUNT(*) FROM tickets").fetchdf())
print(con.execute("DESCRIBE tickets").fetchdf())





# View 1: SLA compliance summary by priority
con.execute("""
    CREATE OR REPLACE VIEW sla_by_priority AS
    SELECT
        priority,
        COUNT(*) AS ticket_count,
        ROUND(AVG(resolution_hours), 1) AS avg_resolution_hours,
        ROUND(AVG(sla_breached) * 100, 1) AS breach_rate_pct
    FROM tickets
    GROUP BY priority
    ORDER BY priority
""")

# View 2: SLA compliance by category
con.execute("""
    CREATE OR REPLACE VIEW sla_by_category AS
    SELECT
        Topic_group,
        COUNT(*) AS ticket_count,
        ROUND(AVG(resolution_hours), 1) AS avg_resolution_hours,
        ROUND(AVG(sla_breached) * 100, 1) AS breach_rate_pct
    FROM tickets
    GROUP BY Topic_group
    ORDER BY ticket_count DESC
""")

# View 3: Category x Priority cross-tab (where breaches concentrate)
con.execute("""
    CREATE OR REPLACE VIEW sla_by_category_priority AS
    SELECT
        Topic_group,
        priority,
        COUNT(*) AS ticket_count,
        ROUND(AVG(sla_breached) * 100, 1) AS breach_rate_pct
    FROM tickets
    GROUP BY Topic_group, priority
    ORDER BY Topic_group, priority
""")

print(con.execute("SELECT * FROM sla_by_priority").fetchdf())
print(con.execute("SELECT * FROM sla_by_category").fetchdf())
print(con.execute("SELECT * FROM sla_by_category_priority ORDER BY breach_rate_pct DESC LIMIT 10").fetchdf())


con.execute("COPY sla_by_priority TO 'data/processed/sla_by_priority.csv' (HEADER, DELIMITER ',')")
con.execute("COPY sla_by_category TO 'data/processed/sla_by_category.csv' (HEADER, DELIMITER ',')")
con.execute("COPY sla_by_category_priority TO 'data/processed/sla_by_category_priority.csv' (HEADER, DELIMITER ',')")
con.execute("COPY tickets TO 'data/processed/tickets_full.csv' (HEADER, DELIMITER ',')")


