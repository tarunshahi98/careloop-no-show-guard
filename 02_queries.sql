-- CareLoop Clinics: Schema & Analytics Queries
-- Database: noshowguard

CREATE DATABASE IF NOT EXISTS noshowguard;
USE noshowguard;

-- Clinics master table
CREATE TABLE IF NOT EXISTS clinics (
    clinic_id VARCHAR(5) PRIMARY KEY,
    clinic_name VARCHAR(40) NOT NULL,
    city VARCHAR(20) NOT NULL,
    tier VARCHAR(10),
    latitude DECIMAL(8,4),
    longitude DECIMAL(8,4),
    temp_c DECIMAL(5,1)
);

-- Clean appointments table with foreign key and domain constraints
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id VARCHAR(8) PRIMARY KEY,
    patient_id VARCHAR(8) NOT NULL,
    clinic_id VARCHAR(5) NOT NULL,
    booked_on DATE,
    appt_date DATE,
    age INT CHECK (age BETWEEN 0 AND 120),
    gender VARCHAR(1) CHECK (gender IN ('M', 'F')),
    distance_km DECIMAL(5,1) CHECK (distance_km >= 0.0),
    fee_inr INT CHECK (fee_inr >= 0),
    booking_channel VARCHAR(10) CHECK (booking_channel IN ('App', 'Phone', 'Walk-In')),
    visit_type VARCHAR(12) CHECK (visit_type IN ('Consultation', 'Follow-up')),
    reminder_sent VARCHAR(3) CHECK (reminder_sent IN ('Yes', 'No')),
    prior_no_shows INT CHECK (prior_no_shows >= 0),
    status VARCHAR(10) CHECK (status IN ('Attended', 'No-Show')),
    lead_days INT CHECK (lead_days >= 0),
    no_show INT CHECK (no_show IN (0, 1)),
    FOREIGN KEY (clinic_id) REFERENCES clinics(clinic_id)
);

-- Note: Data is populated via load_data.sql (8 clinics + 990 appointments)

-- 1. Load verification: appointment volume and overall baseline no-show rate
SELECT 
    COUNT(*) AS total_appointments,
    SUM(no_show) AS total_no_shows,
    ROUND(100.0 * SUM(no_show) / COUNT(*), 2) AS overall_no_show_rate_pct
FROM appointments;

-- Clinic record count check
SELECT COUNT(*) AS total_clinics FROM clinics;

-- Primary key constraint check (attempt duplicate insert with existing appointment_id 'A0001')
-- INSERT INTO appointments VALUES ('A0001','P999','C01','2025-01-01','2025-01-05',30,'M',5.0,500,'App','Consultation','Yes',0,'Attended',4,0);
-- Duplicate-key error text:
-- MySQL: ERROR 1062 (23000): Duplicate entry 'A0001' for key 'appointments.PRIMARY'
-- SQLite: UNIQUE constraint failed: appointments.appointment_id

-- 2. No-show rate by clinic (minimum 95 appointments)
SELECT 
    c.clinic_name,
    c.city,
    COUNT(*) AS appointments,
    SUM(a.no_show) AS no_shows,
    ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments a
JOIN clinics c ON a.clinic_id = c.clinic_id
GROUP BY c.clinic_id, c.clinic_name, c.city
HAVING COUNT(*) >= 95
ORDER BY no_show_pct DESC;

-- No-show rate across all 8 clinics (including smaller clinics)
SELECT 
    c.clinic_name,
    c.city,
    COUNT(*) AS appointments,
    SUM(a.no_show) AS no_shows,
    ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments a
JOIN clinics c ON a.clinic_id = c.clinic_id
GROUP BY c.clinic_id, c.clinic_name, c.city
ORDER BY no_show_pct DESC;

-- 3. Lead time cohort analysis
SELECT 
    CASE 
        WHEN lead_days BETWEEN 1 AND 7 THEN '1-7'
        WHEN lead_days BETWEEN 8 AND 14 THEN '8-14'
        WHEN lead_days BETWEEN 15 AND 30 THEN '15-30'
        WHEN lead_days >= 31 THEN '31+'
        ELSE 'Invalid'
    END AS lead_time_band,
    COUNT(*) AS appointments,
    SUM(no_show) AS no_shows,
    ROUND(100.0 * SUM(no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments
GROUP BY lead_time_band
ORDER BY MIN(lead_days);

-- 4. Month-over-month trend analysis (change in percentage points)
WITH monthly AS (
    SELECT 
        SUBSTR(appt_date, 1, 7) AS month,
        COUNT(*) AS appointments,
        ROUND(100.0 * SUM(no_show) / COUNT(*), 1) AS no_show_pct
    FROM appointments
    GROUP BY SUBSTR(appt_date, 1, 7)
)
SELECT 
    month,
    appointments,
    no_show_pct,
    ROUND(no_show_pct - LAG(no_show_pct) OVER (ORDER BY month), 1) AS change_pp
FROM monthly
ORDER BY month;

-- 5. Clinic ranking within city and comparison against network benchmark
WITH clinic_rates AS (
    SELECT 
        c.clinic_name,
        c.city,
        ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
    FROM appointments a
    JOIN clinics c ON a.clinic_id = c.clinic_id
    GROUP BY c.clinic_id, c.clinic_name, c.city
)
SELECT 
    city,
    clinic_name,
    no_show_pct,
    RANK() OVER (PARTITION BY city ORDER BY no_show_pct DESC) AS rank_in_city,
    CASE 
        WHEN no_show_pct > (SELECT 100.0 * SUM(no_show) / COUNT(*) FROM appointments) THEN 'above'
        WHEN no_show_pct < (SELECT 100.0 * SUM(no_show) / COUNT(*) FROM appointments) THEN 'below'
        ELSE 'equal'
    END AS vs_overall
FROM clinic_rates
ORDER BY city, rank_in_city;
