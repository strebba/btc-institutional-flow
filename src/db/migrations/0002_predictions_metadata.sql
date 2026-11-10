-- Flag di ricostruzione point-in-time sulle predizioni.
--
-- Nullable: le coorti live già scritte restano NULL. Le righe prodotte da
-- scripts/reconstruct_cohort.py portano un JSON con "reconstructed": true
-- (e gli input degradati, se il GEX del giorno non è ricostruibile).
-- La source resta dealer_flow, così la coorte occupa lo slot del giorno
-- (unique su date+source+target+horizon) senza essere un'altra source.

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS metadata TEXT;
