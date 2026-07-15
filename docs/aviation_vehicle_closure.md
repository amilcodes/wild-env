# Aviation vehicle closure

## Scope

This work answers two different questions:

1. Which aircraft and uncrewed systems are actually used in U.S. wildfire
   operations?
2. Which of their parameters are strong enough to constrain a simulator?

The first question can be answered from current public agency material. The
second ultimately requires controlled aircraft documents and operational data.
The implementation keeps those evidence levels separate.

## Operational baseline

The federal wildfire fleet is heterogeneous and largely contracted. NIFC
reports that federal agencies manage roughly 200–300 aircraft and that almost
all aviation resources are privately contracted. USFS leads large and very
large airtanker and multi-engine scooper contracting; BLM manages the SEAT
fleet and most single-engine scoopers. The national categories include
2,000–4,000 gallon large airtankers, SEATs up to 800 gallons, water scoopers,
helicopters, aerial-supervision aircraft, mapping aircraft, transports, and
smokejumper aircraft.

The reference fleet uses CAL FIRE models for the named crewed aircraft because
CAL FIRE publishes a current, model-specific state fleet and nominal
specifications. Its current program lists S-2T and C-130 airtankers, S-70i
FIREHAWK and UH-1H helicopters, OV-10 tactical aircraft, and King Air 200
intelligence platforms. The selection is representative, traceable, and
coherent as one operating system. It is not intended to enumerate every
federal contract model.

Current interagency UAS orders are organized around:

- situational awareness, infrared sensing, and small-area mapping;
- aerial ignition; and
- contracted Type 1 UAS support to large fires.

Orders include UAS personnel and dispatch coordination. Small fire UAS should
therefore be represented as remotely piloted or supervised-autonomy resources,
depending on the aircraft, mission software, and approval. The current
government material does not establish operational autonomous suppressant
delivery by coordinated aircraft swarms. The simulator's water/retardant swarm
is a research system, whereas the reference UAS profiles represent existing
sensing and ignition roles.

Primary operational sources:

- [NIFC aircraft overview](https://www.nifc.gov/resources/aircraft)
- [NIFC airtanker categories](https://www.nifc.gov/resources/aircraft/airtankers)
- [NIFC helicopter roles and types](https://www.nifc.gov/resources/aircraft/helicopters)
- [2026 interagency aircraft mobilization standards](https://www.nifc.gov/sites/default/files/NICC/3-Logistics/Reference%20Documents/Mob%20Guide/2026/2026%20National%20Interagency%20Standards%20for%20Resource%20Mobilization_2_0.pdf)
- [Interagency fire UAS ordering](https://uas.nifc.gov/interagency-fire-uas-ordering)
- [CAL FIRE aviation fleet](https://www.fire.ca.gov/what-we-do/fire-protection/aviation-program)

## Selected profiles

| Profile | Simulator role | Operational control | Public nominal data |
|---|---|---|---|
| CAL FIRE S-2T | Retardant | Crewed | 1,200 gal; speed, range, endurance |
| CAL FIRE C-130H | Retardant | Crewed | 4,000 gal; speed, range, endurance |
| CAL FIRE S-70i FIREHAWK | Water | Crewed | 1,000 gal; cruise, endurance |
| CAL FIRE UH-1H | Water | Crewed | 360 gal tank; cruise, endurance |
| CAL FIRE OV-10A | Air tactical/sensor | Crewed | speed and endurance |
| CAL FIRE King Air 200 | IR/intelligence | Crewed | speed and endurance |
| Skydio X10 | Sensor | Supervised autonomy | flight time, speed, ceiling, wind |
| Parrot ANAFI USA | Sensor | Remotely piloted | flight time, speed, ceiling, wind |
| Freefly Alta X | Aerial ignition/sensor | Remotely piloted | payload and flight-time examples |

The machine-readable catalog is
`configs/aviation/us_wildfire_reference_fleet_v1.json`. Every simulator
parameter has one of four bases:

- `published`: directly stated by the source;
- `unit_conversion`: a direct conversion of a published value;
- `role_mapping`: a simulator representation of the operational role; or
- `modeling_assumption`: a value selected for research use.

The materialized reference experiment is
`configs/aviation/us_wildfire_reference_operations.yaml`. It contains two S-2T
resources, one C-130H, one FIREHAWK, one UH-1H, one OV-10A, one King Air, one
Skydio X10, and one Alta X. Service sites include a retardant base, helibase,
dip site, and UAS launch site.

## Simulator coupling

`ResourceSpec` now retains:

- the catalog profile identifier;
- the performance evidence grade;
- operational control/autonomy level;
- operational roles; and
- optional tactical performance- and delivery-surface paths.

The aviation dynamics evaluate route length, wind components, density altitude,
terrain clearance, service-site geometry, payload, reserve endurance, airspace
volumes, and a vehicle performance surface when one is supplied. Action masks
remove infeasible tasks before a policy samples an assignment.

The reference configuration passes schema checks and instantiates all nine
resources in the canonical simulator. The initial observation exposes valid
task masks for every resource. Catalog tests also reject duplicate keys,
unsupported evidence labels, missing required simulator parameters, and any
field-grade mobility claim without a performance surface or delivery claim
without a delivery surface.

## Present evidence level

All nine mobility profiles remain `scenario_assumption`. This is intentional.
The S-2T delivery subsystem is separately graded `engineering_validated` and
uses the USDA Forest Service's controlled 2006 drop-test table. At coverage
level 3, the measured 595-foot line replaces the prior generic 650-meter line.
Current tail-level configuration continuity is not public, so this closes a
research delivery reference rather than the aircraft's field qualification.
The complete acquisition result and per-domain closure matrix are in
[`aviation_evidence_acquisition.md`](aviation_evidence_acquisition.md). The
submission-ready CAL FIRE request scopes are in
[`aviation_records_request.md`](aviation_records_request.md).

Public agency fleet pages provide useful nominal capacity, speed, endurance,
and role data. They do not provide a complete approved schedule for:

- allowable payload versus pressure/density altitude and temperature;
- takeoff, landing, hover, or one-engine-inoperative performance;
- aircraft weight, balance, fuel state, and installed firefighting equipment;
- climb, cruise, reserve, and diversion performance;
- crosswind and visibility limits for the applicable mission approval;
- bucket, tank, snorkel, scoop, or retardant-system delivery envelopes;
- runway, helibase, dip-site, or scooping-water restrictions; or
- crew duty, maintenance, inspection, dispatch, and turnaround constraints.

The 2026 mobilization standard confirms that density/high-altitude performance,
aircraft type, special capability, and installed equipment are aircraft
selection factors. A field-use model therefore needs the authorized source
documents for the exact aircraft, engine, installed equipment, and operating
approval.

## Required closure package

For each aircraft configuration, obtain with the operator's authorization:

1. approved aircraft flight manual and supplements;
2. current weight-and-balance and equipment list;
3. approved performance schedules across altitude, temperature, and weight;
4. firefighting tank/bucket/scoop system limitations and drop test data;
5. base, runway, helibase, dip-site, and water-source constraints;
6. fuel planning, reserve, maintenance, and crew-duty rules; and
7. representative dispatch, loading, refill, queue, and mission-cycle logs.

These inputs should be compiled into a versioned performance surface with
density altitude and payload fraction as minimum axes. Helicopters also need
hover in/out of ground effect and gross-weight closure; airtankers need
runway/takeoff/landing and delivery-system envelopes; scoopers need water-run
geometry and wave/wind constraints.

The catalog promotes a profile to `flight_manual` or
`engineering_validated` only when that surface is attached. Until then, it is
appropriate for algorithm research and sensitivity analysis, not dispatch,
flight planning, or field safety decisions.
