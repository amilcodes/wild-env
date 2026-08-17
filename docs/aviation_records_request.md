# Exact-configuration aviation records request package

## Purpose and submission route

The public evidence audit identifies the exact records needed to replace
research assumptions with configuration-specific envelopes for the four CAL
FIRE suppression aircraft modeled by Aeolus. CAL FIRE directs requests to its
[GovQA Public Records Center](https://calfire.govqa.us/WEBAPP/_rs/SupportHome.aspx);
its [records guidelines](https://www.fire.ca.gov/about/resources/california-public-records)
also permit written requests to the CAL FIRE Legal Office, PO Box 944246,
Sacramento, CA 94244-2460.

These requests should be filed separately. That keeps each search focused,
allows CAL FIRE to route it to the relevant custodian, and prevents one exempt
or vendor-controlled record from delaying the other responses. Electronic
native files are preferred. CAL FIRE states that a request can be anonymous,
but an email address or other contact channel is still required for
correspondence and release. No request has been submitted from this repository.

The material is requested for non-operational wildfire-simulation research.
The simulator will not treat a record as field authorization, dispatch
approval, or pilot guidance.

The portal requires an account tied to an email address before its request form
can be opened. That is the one missing input for electronic submission.

## Common handling language

Append the following paragraph to each request:

> Please provide records in their existing electronic format, including
> spreadsheets or tabular exports where maintained in that form. This request
> does not seek personal information, security credentials, radio keys,
> personnel schedules, individual pilot records, maintenance discrepancies, or
> tail locations. If a record contains exempt material, please provide every
> reasonably segregable non-exempt portion and identify the legal basis for
> each withholding. If the requested terminology does not match the
> Department's record titles, please identify the responsible office or record
> series and assist in narrowing the request. A rolling response is welcome.

## Request A — S-2T flight and retardant-system configuration

> Under the California Public Records Act, I request current records governing
> the CAL FIRE Marsh Aviation S-2T airtanker configuration, limited to: (1) the
> portions of the CAL FIRE 8300 Aviation Handbook and Fixed-Wing Flight
> Standards that prescribe S-2T loading, landing loaded, fuel reserve,
> environmental or density-altitude limits, weight and balance, retardant-drop
> altitude/airspeed, and constant-flow tank/controller operation; (2) the
> current approved aircraft-flight-manual supplements, operating limitations,
> or equivalent Department summary tables for those domains; (3) the current
> tank/controller model, software or calibration identifier, and approved
> coverage-level schedule by aircraft or configuration group; and (4) records
> establishing whether the delivery configuration evaluated in USDA Forest
> Service report 0657-2848-MTDC, *Ground Pattern Performance of the CDF S-2T
> Turbo Tracker Airtanker* (December 2006), remains applicable to the present
> fleet. A tail-by-tail equipment list is useful, but aircraft registration,
> serial number, or a Department configuration-group identifier is sufficient.
> The requested date range is January 1, 2024 through the date of search, except
> for still-current manuals or approval records issued earlier.

Minimum usable response: a current configuration crosswalk, the operative
drop-controller schedule, and confirmation or rejection of the 2006 drop-test
applicability.

## Request B — C-130H 4,000-gallon RDS engineering envelope

> Under the California Public Records Act, I request current technical records
> for CAL FIRE's ex-USCG HC-130H airtankers equipped with the 4,000-gallon
> retardant delivery system (RDS), limited to: (1) the approved RDS
> aircraft-flight-manual supplement or equivalent operating-limitations
> document; (2) RDS installed weight, center-of-gravity, drag, and usable-load
> data; (3) takeoff, landing, climb, engine-out, fuel-reserve, and mission-range
> schedules or Department-approved tabular exports across pressure altitude,
> temperature, gross weight, and retardant load; (4) RDS door, flow-rate,
> coverage-level, drop-altitude, and drop-airspeed limits; and (5) Interagency
> Airtanker Board or successor qualification summaries and controlled
> deposition-test tables for this exact 4,000-gallon installation. This request
> concerns the CAL FIRE RDS and does not seek MAFFS records. The requested date
> range is January 1, 2020 through the date of search, including earlier records
> incorporated into a current approval basis.

Minimum usable response: RDS installed mass/drag data, the high/hot
payload-performance table, and coverage-level deposition results.

## Request C — S-70i CAL FIRE HAWK performance and tank delivery

> Under the California Public Records Act, I request current technical records
> for the CAL FIRE S-70i CAL FIRE HAWK configuration, limited to: (1) a
> configuration or installed-options list sufficient to reproduce aircraft
> performance, including belly tank, snorkel, hoist, crew, mission equipment,
> and basic operating weight; (2) Department-approved Sikorsky iFly exports,
> reviewed performance tables, or equivalent summaries across pressure
> altitude, temperature, gross weight, tank state, and crew/equipment state for
> hover in-ground-effect, hover out-of-ground-effect, takeoff, cruise, reserve,
> and one-engine-inoperative conditions; (3) tank fill-time and usable-water
> data across representative source elevation and aircraft weight; (4) water
> source depth, approach, obstacle-clearance, and hover-power constraints; and
> (5) tank door/flow schedules and controlled water-deposition tests, including
> gridded cup data where retained. The requested date range is January 1, 2022
> through the date of search, including earlier records incorporated into a
> current approval basis.

Minimum usable response: a reviewed iFly case grid for the CAL FIRE build and
the tank-system deposition/operating envelope.

## Request D — UH-1H Super Huey configuration closure

> Under the California Public Records Act, I request current technical records
> for CAL FIRE's modified UH-1H Super Huey configuration, limited to: (1) an
> installed supplemental-type-certificate and equipment crosswalk by aircraft
> or Department configuration group, covering the T53-L-703 conversion, main
> rotor blades, tail boom/tail rotor modifications, fixed tank, and bucket
> provisions; (2) the operative aircraft-flight-manual supplements or
> Department-approved performance tables for those combined modifications;
> (3) basic operating weight and center-of-gravity envelope by configuration
> group; (4) hover in-ground-effect, hover out-of-ground-effect, takeoff,
> payload, cruise, reserve, and directional-control schedules across pressure
> altitude, temperature, and gross weight; and (5) fixed-tank and approved
> bucket delivery-system limits and controlled water-deposition tests. The
> requested date range is January 1, 2022 through the date of search, including
> earlier records incorporated into a current approval basis.

Minimum usable response: the STC/configuration crosswalk plus performance
tables that account for the combined -703, rotor, tail, and tank state. Base
Army T53-L-13 UH-1H charts do not satisfy this request.

## Intake and acceptance procedure

Received files should enter `tmp/aviation_evidence/incoming/` and remain
uncommitted until reviewed for release restrictions. For each file:

1. preserve the original filename, byte stream, response letter, and delivery
   date;
2. compute SHA-256 and add the source, custodian, issue/revision, applicability,
   and restrictions to `configs/aviation/evidence_registry_v1.json`;
3. identify the precise aircraft/configuration and the independent variables
   of every table before transcription;
4. add a regression test against at least one source-table row and a boundary
   test for each envelope axis;
5. advance an evidence domain to `closed` only if the record covers the exact
   current configuration and carries approved-flight-manual or controlled
   engineering authority; and
6. retain uncertainty or interpolation limits in runtime diagnostics. Never
   silently replace an open domain with a base-airframe or MAFFS proxy.

Vendor manuals supplied under a license should not be committed unless the
license permits redistribution. A derived, non-invertible coefficient set may
be versioned only when the license and response terms permit that use and its
provenance remains auditable.
