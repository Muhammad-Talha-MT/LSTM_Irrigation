# Method and experimental protocol

All weather, soil measurements, irrigation labels and farm identities in the
included experiments are synthetic. There are no downloaded farm observations.
The study demonstrates implementation and behavior, not field effectiveness,
yield improvement, water savings relative to expert practice, or a universal
minimum data requirement.

## Physical model and information timing

The modeled period is April 1–May 30 inclusive (60 days), preceded by 30 warmup
days. Dates use 2001 solely as a calendar template. `year` is a synthetic season
identifier, not a historical calendar year. Planting is assumed April 1 in all
experiments. Predictions are made every morning after overnight rain is measured.
Temperature, humidity, wind and solar inputs are **preceding-24-hour summaries**.
Current-day atmospheric demand and next overnight rain enter only as imperfect
forecasts. Actual daytime evapotranspiration is used only in the environment.

The bucket represents plant-available storage S, in mm above wilting point:

- TAW = 1000 (theta_FC − theta_WP) Zr.
- Root depth grows from 0.20 m at planting by 0.011 m/day, capped by each farm's
  maximum (0.75–1.05 m). This is an illustrative early-season growth curve.
- When roots explore additional soil, 0.65 times the increase in TAW is added as
  available water from the newly explored layer. This explicit flux prevents
  root growth from silently creating or destroying water.
- Incoming rain loses a farm-specific runoff fraction; infiltration is also
  limited by a daily capacity. Water above TAW drains immediately.
- Irrigation adds efficiency × gross depth; excess becomes drainage.
- Kc grows from 0.35 to 1.10. Potential ET = Kc × synthetic ETo.
- Actual ET = Ks × potential ET, capped by available water, with
  Ks = min(1, S / (0.45 TAW)) after irrigation.
- End-of-day S = previous S + root-growth water + rain + net irrigation
  − runoff − drainage − actual ET.

TAW and the basic stress concept follow FAO-56. The specific growth curves,
forecast errors, teacher thresholds, daily infiltration approximation, and
weather generator are our experimental assumptions, **not a calibrated FAO-56
implementation**. ETo is generated directly; the code does not implement the
Penman–Monteith equation. Waterlogging duration, oxygen stress, groundwater,
nutrients, spatial soil variability and yield are not modeled. In particular,
immediate drainage makes this a poor test of persistent flooding.

## Measurements, labels and safety checks

Root-zone moisture has independent Gaussian sensor error with standard deviation
0.003 m³/m³. True simulator storage is unavailable to the learned controller and
teacher; both use the same noisy observation. Depletion is derived using measured
soil properties. A field installation would estimate root-zone moisture from
multiple depths and locations rather than a perfect single representative sensor.

The teacher projects depletion as measured current depletion + forecast crop ET
− 0.5 × forecast overnight rain. If it exceeds 0.50 TAW, the teacher requests
refilling toward 0.15 TAW. Both numbers are policy choices. Requests are converted
to gross depth by application efficiency, limited to equipment capacity and
current observed storage deficit, and set to zero below 0.25 TAW depletion.
This last check is also applied to learned recommendations. It is not a guarantee
against bad sensors, unexpected rain after actuation, or incorrect soil properties.

Labels therefore encode a known heuristic, not experimentally established optimal
irrigation. The teacher is an exact label-generating reference and a closed-loop
baseline. No-irrigation is a diagnostic lower-water/higher-stress reference.

## LSTM and training

History: 14 previous days × 14 dynamic features. The current 14 dynamic features
and six static properties enter a separate branch; this lets a fresh moisture
reading affect the recommendation immediately. History excludes the current day.

The dynamic features, in order, are rainfall (mm), measured moisture (m³/m³),
measured depletion fraction, forecast ETo (mm/day), forecast next overnight rain
(mm), previous-24-hour minimum and maximum temperature (°C), relative humidity
(%), wind (m/s), solar radiation (MJ/m²/day), days after planting, rooting depth
(m), previous day's gross irrigation (mm), and crop coefficient (dimensionless).
The static inputs are field capacity and wilting point (m³/m³), infiltration
capacity (mm/day), efficiency (fraction), maximum gross daily depth (mm/day), and
runoff fraction. Farm ID is NOT a learned input.

A one-layer 24-unit LSTM summarizes the history. Its last hidden vector is
concatenated with the 20 current/static features. A 32-unit ReLU layer leads to
two outputs: a binary irrigation logit and conditional depth in [0,30] mm via
a sigmoid. Equipment limits may be lower. At probability >= 0.5, the conditional
amount is requested; otherwise zero is requested. Then the physical guard runs.

Loss = binary cross-entropy + mean squared conditional-depth error / 30² on
positive-label samples. Amount regression is not trained against zero on dry
decision days. Adam uses learning rate 0.003, batch size 128, gradient norm cap 1,
at most 65 epochs, and early stopping after 12 epochs without validation-loss
improvement. The best validation checkpoint is restored. All normalization is
fitted on training data only. The recurrent state resets for each sliding window;
there is no undocumented persistent hidden state between forecasts.

## Splits and comparisons

- Source farms: 0–7. Training seasons: 0–5 (48 farm-seasons, 2,880 decision
  samples). Validation: season 6 (480). Test: seasons 7–8 (960).
- Target farms: 8 (sandy/low capacity) and 9 (slow-infiltrating/high wilting
  point). Their records are never used in source training or normalization.
- For each target separately: seasons 0–1 supply 120 adaptation samples;
  season 2 supplies 60 validation samples; seasons 7–8 supply 120 test samples.
- Transfer comparisons: unchanged source model; output-head-only adaptation
  (LSTM frozen, learning rate 0.001); all-layer adaptation (0.0005); and an
  identically structured model trained from scratch (0.003). Adapted models
  retain source normalization; scratch fits target-training normalization.
- Data size: 1, 3 or 6 source seasons = 480, 1,440 or 2,880 decision samples.
  This means eight farms in each condition, NOT one farm with that many years.
- History length: 7, 14 or 30 days, with the same decision dates and splits.
- MLP: same current/static features, 32-unit head, no recurrent/history branch.
  This is an intentional test of whether history is necessary when current
  root-zone storage is directly measured.
- Ordinary versus extreme-aware training: equal 2,880 training samples. In
  augmentation, half the trajectories contain two unforecast 45–95 mm events
  at randomly chosen dates. Validation is similarly a fixed ordinary/extreme mix.
  Test storm is 80 mm on May 1. Ordinary rain is capped at 35 mm.
- Seeds 11, 22, 33 vary neural initialization and minibatch order. Generated
  farms/weather and splits are fixed. Standard deviations across these seeds
  quantify optimization sensitivity, NOT uncertainty over real farms/climates.

## Metrics and closed-loop test

One-step held-out prediction reports unguarded and guarded depth MAE, guarded
RMSE, irrigation-event F1, and probability Brier score. These use teacher-driven
soil trajectories and measure imitation, not decision utility.

Closed-loop rollouts run each controller on its own evolving soil state with
identical exogenous weather and measurement noise. We report total gross water,
end-of-day depletion-threshold exceedance days, modeled ET shortfall, drainage,
wet-state irrigation events, May 1 irrigation, and total May irrigation. A
`stress_day` means end-of-day depletion > 0.55 TAW; it is a proxy, not a measured
plant injury or yield reduction. Less irrigation alone is not success: it must
be interpreted alongside stress and ET shortfall.

An intentionally stale schedule replays the ordinary-weather LSTM's 60-day
actions into the storm environment without updates or the wet-soil guard. This
is a counterfactual demonstration of failing to revise a schedule, NOT a fair
alternative forecast system. It uses an ordinary-scenario schedule computed
offline and is not presented as a deployable April 1 forecast.

The storm occurs before the May 1 morning decision. If a real storm starts
after a morning decision, a separate rain sensor/actuation interlock must stop
irrigation; once-daily prediction alone cannot respond to an event it has not
yet observed. No model is retrained during these May rollouts.

## References

Allen, R.G., Pereira, L.S., Raes, D., and Smith, M. (1998). *Crop
evapotranspiration: Guidelines for computing crop water requirements.* FAO
Irrigation and Drainage Paper 56. Chapter 8.
https://www.fao.org/4/X0490E/x0490e0e.htm

FAO-56 Chapter 4, reference evapotranspiration (recommended field preprocessing;
not implemented in this synthetic generator):
https://www.fao.org/4/X0490E/x0490e08.htm

Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory.
*Neural Computation*, 9(8), 1735–1780. doi:10.1162/neco.1997.9.8.1735.

PyTorch, LSTM API documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
