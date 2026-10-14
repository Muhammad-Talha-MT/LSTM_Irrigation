# Moving from this experiment to a real farm

The delivered checkpoint is a demonstrator. This document identifies the missing
evidence rather than implying that the synthetic model can simply be deployed.

## Daily inputs and records

Use one record per irrigation zone per decision timestamp. Keep the timestamp
and time zone, planting date and season, field/zone identifier, sensor quality
flags, and the exact weather-forecast issue time. Store missing readings as
missing, not as zero rain or zero soil moisture.

| Measurement | Units and collection |
|---|---|
| Precipitation | mm over the preceding 24 hours; local gauge, with storm intensity at subdaily resolution if possible |
| Soil moisture | m³/m³ from multiple depths spanning the developing roots and representative locations |
| Soil temperature | °C; useful for early development and cold/wet establishment |
| Air temperature | preceding-24-hour min/max °C and forecast temperatures |
| Humidity | % or vapor pressure, with forecast equivalents |
| Wind | m/s, preferably standardized to 2 m |
| Solar radiation | MJ/m²/day or correctly converted W/m² records |
| Forecast precipitation | mm and probability for each issue/lead time, archived as originally issued |
| Reference ET | mm/day derived consistently from weather, e.g. FAO-56 Penman–Monteith |
| Crop condition | cultivar, planting/emergence dates, stage, canopy and observed/estimated rooting depth |
| Soil profile | texture, layer thickness, field capacity, wilting point, infiltration and drainage properties |
| Groundwater/ponding | depth in m, duration of standing water and drainage condition where relevant |
| Applied irrigation | measured gross mm from meter volume / irrigated area; start/stop time and equipment rate |
| Operations | system efficiency, capacity, allowed hours, water allocation and interruptions |
| Outcomes | soil water response, crop stress observations and eventual yield, if claiming yield benefits |

The demonstration reduces some of these to a single effective root-zone state.
It does not currently ingest soil-temperature, groundwater, crop damage or sensor
quality channels. Adding channels requires corresponding local data, architecture
changes and retraining; they are not silently supported by the current checkpoint.

## Label decisions

Historical irrigation is an action, not automatically an optimal target. Have an
agronomist audit labels against soil moisture and operational constraints. A
teacher based on a calibrated water balance can supply initial labels, but a
model trained on it cannot claim to surpass that teacher solely from low MAE.
If learning outcome-optimal scheduling, define the costs of stress, water,
energy, labor and drainage, and use appropriate controlled comparisons; that is
outside this repository's supervised-imitation scope.

## Data sufficiency and validation

Start with multiple wet, normal and dry seasons and multiple soil/equipment
settings, rather than assuming a universal five- or ten-year rule. Count actual
irrigation and extreme-event examples. Plot learning curves on whole held-out
seasons. Hold out farms for spatial transfer, and storm/recovery episodes for
extreme-event testing. Compare the LSTM with a calibrated water balance, a
current-state MLP and operational practice on the same available information.

Test target-farm adaptation against leaving the model unchanged. Keep local
adaptation, validation and final evaluation periods separate. Validate uncertainty
and define an out-of-range/fault fallback before allowing actuation.

Begin with monitored recommendations, then prospective paired zones or a suitable
field-trial design. Evaluate stress and measured crop response alongside water
volume. Add rain-based shutdown independent of the daily neural prediction.

## Direct reuse of code

`src.model.fit` accepts arrays `(history, current_plus_static, target, ids)`.
`src.simulator.DYNAMIC` and `STATIC` define the exact input order. To use real
data, build these arrays from aligned observations and audited target decisions;
do not feed simulator truth columns (`true_theta`, `true_eto_mm`, etc.) to the
model. Fit normalization on the training split only, retain archived forecasts,
and respect the preceding-24-hour timing of meteorological summaries.

The generator-specific `sequences` helper assumes 30 warmup records plus 60
decision records per farm-season. A general field-data loader must instead check
timestamps, missing days and crop stage; silently passing irregular rows to this
helper would make an invalid sequence.
