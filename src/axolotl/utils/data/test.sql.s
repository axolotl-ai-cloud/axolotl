WITH unique_addresses AS (
SELECT DISTINCT
v.pid,
v.hhid,
left(soundex(v.fname), 2) as fname_short_soundex,
concat_ws('\n', v.address, v.city, v.state, v.zip) as full_address
FROM
    narrative_data_collaboration_block.shared.verisk_tci AS v
)

SELECT
    code.narrative_id_encode(v.pid, 'aa4UIQ') as block_id,
    'individual' as block_id_type,
    code.narrative_id_encode(hash(concat(v.fname_short_soundex, f.value)), 'aa4UIQ') as join_id,
    'address_with_name' as join_id_type,
    abs(hash(code.narrative_id_encode(v.pid, 'aa4UIQ'))) % 100 as bucket,
    current_timestamp as updated_at
FROM
    unique_addresses AS v,
    LATERAL FLATTEN(INPUT => CODE.ADDRESS_HASHES(v.full_address)) AS f
UNION ALL
SELECT
    code.narrative_id_encode(v.hhid, 'aa4UIQ') as block_id,
    'household' as block_id_type,
    code.narrative_id_encode(hash(f.value), 'aa4UIQ') as join_id,
    'address' as join_id_type,
    abs(hash(code.narrative_id_encode(v.hhid, 'aa4UIQ'))) % 100 as bucket,
    current_timestamp as updated_at
FROM
    unique_addresses AS v,
    LATERAL FLATTEN(INPUT => CODE.ADDRESS_HASHES(v.full_address)) AS f
UNION ALL
SELECT
code.narrative_id_encode(v.pid, 'aa4UIQ') as block_id,
'individual' as block_id_type,
code.narrative_id_encode(sha2_hex(
CASE
        WHEN LOWER(SPLIT_PART(TRIM(v.email), '@', 2)) = 'gmail.com' THEN
            LOWER(
                SPLIT_PART(
                    REGEXP_REPLACE(SPLIT_PART(TRIM(v.email), '@', 1), '\\.', ''),
                    '+',
                    1
                )
            ) || '@gmail.com'
        ELSE
            LOWER(TRIM(v.email))
    END, 256), 'aa4UIQ') AS join_id,
'sha_256_email' AS join_id_type,
abs(hash(code.narrative_id_encode(v.pid, 'aa4UIQ'))) % 100 as bucket,
current_timestamp as updated_at
FROM
narrative_data_collaboration_block.shared.verisk_tci AS v
WHERE v.email IS NOT NULL
UNION ALL
select code.narrative_id_encode(v.pid, 'aa4UIQ') as block_id,
'individual' as block_id_type,
code.narrative_id_encode(sha2_hex(concat('+1', v.phone), 256), 'aa4UIQ') as join_id,
'sha_256_phone' as join_id_type,
abs(hash(code.narrative_id_encode(v.pid, 'aa4UIQ'))) % 100 as bucket,
current_timestamp as updated_at
from narrative_data_collaboration_block.shared.verisk_tci AS v
where len(v.phone) = 10 AND v.phone_confidence_code = 'H' AND v.phone IS NOT NULL
