from typing import List, Dict
from agents.executor import executor
from state.research_state import PlanStep, ResearchState


def test_executor_agent():
    """Tests the executor agent end-to-end on a simple example plan step."""

    # Define a basic plan with one research goal
    plan: List[PlanStep] = [
        {
            "id": "s1",
            "goal": "Find publicly documented hiking trails in Washington State with total trail length between 4.0 and 6.0 miles (one-way or round-trip explicitly noted), and return for each: trail name, reported length, trailhead coordinates or address, elevation gain, and source URL.",
            "method": "search",
            "risk": "low",
            "produces_entities": ["candidate_trails"],
            "requires_entities": [],
            "expanded_goal": "Find publicly documented hiking trails in Washington State with total trail length between 4.0 and 6.0 miles (one-way or round-trip explicitly noted), and return for each: trail name, reported length, trailhead coordinates or address, elevation gain, and source URL.",
        },
        {
            "id": "s2",
            "goal": "For each candidate_trail, determine driving time by car from a central Seattle point (e.g., Seattle City Hall or downtown Seattle) to the trailhead using a mapping service or travel-time estimates appropriate for winter conditions, and produce a list of trails with estimated winter driving time.",
            "method": "analysis",
            "risk": "medium",
            "produces_entities": ["filtered_by_driving_time"],
            "requires_entities": ["candidate_trails"],
            "expanded_goal": "For each candidate_trail, determine driving time by car from a central Seattle point (e.g., Seattle City Hall or downtown Seattle) to the trailhead using a mapping service or travel-time estimates appropriate for winter conditions, and produce a list of trails with estimated winter driving time.\n\nUse the following context when executing this step:\n\nContext for entity type candidate_trails:\n1. Johnson Mountain\n2. Jungle Creek\n3. Rye Creek to Camp Lake (Snowshoe)\n4. Spruce Railroad Trail\n5. Mud Mountain Dam - Rim Trail\n6. Skookum Flats\n7. Wapta Falls\n8. Kicking Horse Fire Road\n9. Emerald Lakeshore\n10. Tally-Ho\n11. Takakkaw Falls\n12. Centennial\n13. Yoho Valley\n14. Great Divide (to Lake Louise)\n15. Ross Lake\n16. Paget Lookout\n17. Sherbrooke Lake\n18. Emerald Basin\n19. Emerald River\n20. Yoho Glacier Moraine\n21. Little Yoho Valley\n22. Hoodoos\n23. Mt. Hunter Lower Lookout\n24. Mt. Hunter Upper Lookout\n25. Hamilton Lake\n26. Ottertail Valley\n27. Yoho Lake\n28. Tocher Ridge\n29. Emerald Triangle\n30. Iceline via Little Yoho\n31. Iceline via Celeste Lake\n32. Whaleback\n33. Bear Lake Trail (Bear Lake)\n34. a 4 mile (6.4 km) trail with 1,000 ft (304 m) elevation gain\n35. a 6 mile out-and-back\n36. a 4 mile out-and-back\n37. Bristlecone Loop Trail\n38. Wenatchee Foothills Trails\n",
        },
        {
            "id": "s3",
            "goal": "For trails in filtered_by_driving_time that are within 90 minutes, verify winter vehicle and trailhead access by checking WSDOT road status, USFS/State Park/County road closure pages, and trailhead parking pages to confirm whether the trailhead is reachable by car in winter; mark each trail as 'winter-accessible' or not with source URLs.",
            "method": "search",
            "risk": "medium",
            "produces_entities": ["winter_accessible_trails"],
            "requires_entities": ["filtered_by_driving_time"],
        },
        {
            "id": "s4",
            "goal": "For each winter_accessible_trail, collect objective winter-safety indicators: Washington Avalanche Center advisory references for the area, presence of avalanche-prone terrain on the route, trail steepness/elevation profile (sections >30 degrees), official trail difficulty/class, and records of winter incidents from authoritative sources; produce a safety-assessment record per trail.",
            "method": "search",
            "risk": "medium",
            "produces_entities": ["safety_checked_trails"],
            "requires_entities": ["winter_accessible_trails"],
        },
        {
            "id": "s5",
            "goal": "Verify that each safety_checked_trail is located in a mountain setting (e.g., within named mountain range or on a mountain/peak) by checking topo maps, USGS/USFS descriptions, or official park descriptions and record the mountain/range name and trail elevation extremes.",
            "method": "search",
            "risk": "low",
            "produces_entities": ["mountain_trails"],
            "requires_entities": ["safety_checked_trails"],
        },
        {
            "id": "s6",
            "goal": "From mountain_trails, produce a final list of trails that meet all constraints (within 90 minutes from Seattle, car-accessible in winter, 4–6 miles long, located in mountains, and without objective winter-safety disqualifiers), including for each the name, trailhead coordinates, length, elevation gain, estimated winter driving time, safety notes, and source links.",
            "method": "analysis",
            "risk": "low",
            "produces_entities": ["final_trail_list"],
            "requires_entities": ["mountain_trails"],
        },
    ]

    evidence_store = [
        [
            "- Johnson Mountain\n  - Region: Snoqualmie Region > Salmon La Sac/Teanaway\n  - Length: 5.4 miles, roundtrip\n  - Elevation gain: 1,600 feet\n  - Highest point: 5,220 feet\n  - Rating: Average rating 3.00 (1 vote)\n  - Tags: Dogs allowed on leash; Mountain views; Ridges/passes; Summits; Wildflowers/Meadows\n  - Status/notes: 12.19.25 - Trailhead inaccessible due to road closures of 9737 (N. Fork Teanaway), 9701 (Jungle Creek Rd.) and 9737 (along Stafford Creek).\n  - Description (as given): Mostly exposed hike passing through wildflowers and a burn area to reach the Johnson Mountain summit with views of the Enchantments, Rainier, the Teanaway River Valley, and wind farms of eastern Washington.\n\n- Jungle Creek\n  - Region: Snoqualmie Region > Salmon La Sac/Teanaway\n  - Length: 4.0 miles, one-way\n  - Elevation gain: 1,500 feet\n  - Highest point: 4,500 feet\n  - Rating: Average rating 2.60 (5 votes)\n  - Tags: Dogs allowed on leash; Mountain views; Ridges/passes; Wildflowers/Meadows\n  - Status/notes: 12.19.25 - Trailhead inaccessible due to road closures of 9737 (N. Fork Teanaway), 9701 (Jungle Creek Rd.) and 9737 (along Stafford Creek).\n  - Description (as given): Jungle Creek Trail is a 4-mile trail from FS 9701/Jungle Creek Road to Johnson Meadow Trail #1383. The saddle at 2.1 miles from the trailhead is the highest point (4,500 ft). Spring flowers and views of peaks in the Stuart Range at the saddle.\n\n- Rye Creek to Camp Lake (Snowshoe)\n  - Region: Snoqualmie Region > Cle Elum Area\n  - Length: 5.6 miles, roundtrip\n  - Elevation gain: 250 feet\n  - Highest point: 2,750 feet\n  - Rating: Average rating 0.00 (0 votes)\n  - Tags: Dogs allowed on leash; Lakes; Rivers\n  - Status/notes: 12.19.25 - Trail inaccessible due to road closures of 9737 (N. Fork Teanaway), 9701 (Jungle Creek Rd.) and 9737 (along Stafford Creek).\n  - Description (as given): Snowshoe route follows snow-covered forest roads to Camp Lake in the Teanaway Community Forest.\n\n- Spruce Railroad Trail\n  - Region: Olympic Peninsula > Northern Coast\n  - Length: 5.0 miles, one-way\n  - Elevation gain: 250 feet\n  - Highest point: 700 feet\n  - Rating: Average rating 3.87 (39 votes)\n  - Tags: Dogs allowed on leash; Good for kids; Lakes; Mountain views\n  - Status/notes: 12.16.25 - Trail closed due to major landslide.\n  - Description (as given): Scenic, historic hike along the shores of Lake Crescent; microclimate warmer/drier than nearby areas.\n\n- Mud Mountain Dam - Rim Trail\n  - Region: Mount Rainier Area > Chinook Pass - Hwy 410\n  - Length: 4.0 miles, roundtrip\n  - Elevation gain: 80 feet\n  - Highest point: 1,300 feet\n  - Rating: Average rating 2.80 (10 votes)\n  - Tags: Dogs allowed on leash; Fall foliage; Good for kids; Mountain views; Rivers; Wildflowers/Meadows; Wildlife\n  - Status/notes: 12.17.25 - Trailhead inaccessible due to washout on State Route 410 past Enumclaw. (Access through the washout restricted to Greenwater locals and official responders.)\n  - Description (as given): Located on the White River near Mount Rainier; described as a gentle trail along the White River.\n\n- Skookum Flats\n  - Region: Mount Rainier Area > Chinook Pass - Hwy 410\n  - Length: 4.6 miles, roundtrip\n  - Elevation gain: 525 feet\n  - Highest point: 2,515 feet\n  - Rating: Average rating 3.59 (29 votes)\n  - Tags: Dogs allowed on leash; Good for kids; Mountain views; Old growth; Rivers; Waterfalls; Wildflowers/Meadows; Wildlife\n  - Status/notes: 12.17.25 - Trailhead inaccessible due to washout on State Route 410 past Enumclaw. (Access through the washout restricted to Greenwater locals and official responders.)\n  - Description (as given): Gentle walk through shaded forest near the White River. Skookum Falls is a 4.6-mile round trip destination; south trailhead allows choosing segments of an 8.5-mile one-way route.",
            "Wapta Falls — 2.2 km — one-way\n\nKicking Horse Fire Road — 6.9 km — one-way\n\nEmerald Lakeshore — 5.2 km — round trip (loop)\n\nTally-Ho — 3.2 km — one-way\n\nTakakkaw Falls — 0.9 km — one-way\n\nCentennial — 1 km — one-way\n\nYoho Valley — Up to 6.4 km — one-way\n\nGreat Divide (to Lake Louise) — 9.8 km — one-way\n\nRoss Lake — 2.9 km — one-way\n\nPaget Lookout — 3.5 km — one-way\n\nSherbrooke Lake — 4.3 km — one-way\n\nEmerald Basin — 4.8 km — one-way\n\nEmerald River — 9 km — one-way\n\nYoho Glacier Moraine — 8.6 km — one-way\n\nLittle Yoho Valley — 9.8 km — one-way\n\nHoodoos — 3.2 km — one-way\n\nMt. Hunter Lower Lookout — 3.5 km — one-way\n\nMt. Hunter Upper Lookout — 5.7 km — one-way\n\nHamilton Lake — 5.1 km — one-way\n\nOttertail Valley — 14.5 km — one-way\n\nYoho Lake — 4.9 km — one-way\n\nTocher Ridge — 17.9 km — one-way\n\nEmerald Triangle — 18.8 km — round trip (loop)\n\nIceline via Little Yoho — 20.3 km — round trip (loop)\n\nIceline via Celeste Lake — 17.4 km — round trip (loop)\n\nWhaleback — 20.5 km — round trip (loop)",
            "- Trail name: Bear Lake Trail (Bear Lake)\n- Location: Black River Wild Forest, Adirondack Park, near McKeever\n- Trailhead: Pull off on Wolf Lake Landing Road (also reachable from the McKeever Trailhead parking area/kiosk)\n- Trailhead coordinates: N43 36.628  W75 04.127",
            "- Definition\n  - Elevation gain: the total vertical distance climbed during a hike.\n\n- Cumulative elevation example\n  - If you start at sea level, climb 500 ft, descend 200 ft, then climb another 300 ft, your elevation gain = 500 ft + 300 ft = 800 ft.\n\n- Calculating elevation gain per mile\n  - Divide total elevation gain by total trail distance.\n  - Example: a 4 mile (6.4 km) trail with 1,000 ft (304 m) elevation gain → 1,000 ÷ 4 = 250 ft (76 m) per mile.\n\n- Out-and-back trails (conversion for per-mile calculation)\n  - For out-and-back trails, first divide the total mileage by two (use one-way mileage) before dividing the elevation gain by mileage.\n  - Examples: a 6 mile out-and-back → use 3 miles for the per-mile calculation; a 4 mile out-and-back → use 2 miles.\n\n- Elevation grade (steepness)\n  - Grade is expressed as a percentage and represents inclination (positive) or declination (negative).\n  - Higher percentage = steeper incline (e.g., 50% steeper than 20% steeper than 10%).\n  - Negative percentages indicate downhill.\n  - The perceived steepness depends on distance (e.g., a 30% grade over a mile is much shallower than the same grade over 200–300 ft).",
            "- Guidance: Identify an official source for the trail (city/state/federal agencies, land trusts, or land managers). An ideal source provides: Trail Names; Trail Access & Use; Trail Operator.\n\n- Example official references (trail name, source/operator, source URL):\n  - Bristlecone Loop Trail — National Park Service — https://www.nps.gov/thingstodo/bristlecone-loop-trail.htm\n  - Wenatchee Foothills Trails — Chelan‑Douglas Land Trust — https://www.cdlandtrust.org/trails-access/trails/wenatchee-foothills-trails\n\n- Recommended minimum tags for an official trail in OpenStreetMap:\n  - highway=path\n  - name=[Name of Trail]\n  - operator=* and/or operator:wikidata=*\n  - Access tags for allowed modes of travel (example: foot=designated)\n\n- Bristlecone Loop Trail — example OpenStreetMap tags added from the official NPS reference:\n  - Allowed Access: foot=designated, motor_vehicle=no, bicycle=no, dog=no\n  - name=Bristlecone Loop Trail\n  - operator=National Park Service\n  - informal=no",
        ]
    ]

    entities = {
        "candidate_trails": [
            "Johnson Mountain",
            "Jungle Creek",
            "Rye Creek to Camp Lake (Snowshoe)",
            "Spruce Railroad Trail",
            "Mud Mountain Dam - Rim Trail",
            "Skookum Flats",
            "Wapta Falls",
            "Kicking Horse Fire Road",
            "Emerald Lakeshore",
            "Tally-Ho",
            "Takakkaw Falls",
            "Centennial",
            "Yoho Valley",
            "Great Divide (to Lake Louise)",
            "Ross Lake",
            "Paget Lookout",
            "Sherbrooke Lake",
            "Emerald Basin",
            "Emerald River",
            "Yoho Glacier Moraine",
            "Little Yoho Valley",
            "Hoodoos",
            "Mt. Hunter Lower Lookout",
            "Mt. Hunter Upper Lookout",
            "Hamilton Lake",
            "Ottertail Valley",
            "Yoho Lake",
            "Tocher Ridge",
            "Emerald Triangle",
            "Iceline via Little Yoho",
            "Iceline via Celeste Lake",
            "Whaleback",
            "Bear Lake Trail (Bear Lake)",
            "a 4 mile (6.4 km) trail with 1,000 ft (304 m) elevation gain",
            "a 6 mile out-and-back",
            "a 4 mile out-and-back",
            "Bristlecone Loop Trail",
            "Wenatchee Foothills Trails",
        ]
    }

    # Construct a minimal initial state
    initial_state: ResearchState = {
        "user_query": "Gather a list of good winter hiking trails in Washington State. Must be within 90 mins from seattle, accessible by car in the winter, between 4-6 miles long, not too dangerous, in the mountains",
        "clarified_query": "Gather a list of good winter hiking trails in Washington State. Must be within 90 mins from seattle, accessible by car in the winter, between 4-6 miles long, not too dangerous, in the mountains",
        "clarity_score": 0.9,
        "clarification_needed": False,
        "research_brief": None,
        "plan": plan,
        "current_step_idx": 1,
        "replan_request": None,
        "entities": entities,
        "evidence_store": evidence_store,
        "failed_steps": [],
        "supervisor_decision": None,
        "termination_reason": None,
        "replan_count": 0,
        "max_replans": 2,
    }

    # Run the executor
    print("Running executor...")
    updated_fields = executor(initial_state)

    # Print results
    print("=== Executor Test Output ===")
    evidence_list = updated_fields.get("evidence_store", [])
    with open("src/data/executor_result.txt", "w", encoding="utf-8") as f:
        print(updated_fields, file=f)
    for i, evidence_group in enumerate(evidence_list):
        print(f"\nStep {i} Evidence:")
        for j, evidence in enumerate(evidence_group):
            print(f"  [{j}]")
            if isinstance(evidence, dict):
                for k, v in evidence.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {evidence}")

    # Optional: basic assertions
    assert len(evidence_list) > 0, "No evidence returned"
    assert isinstance(evidence_list[0], list), "Evidence not grouped by step"
    print("\nTest passed")


def main():
    test_executor_agent()


if __name__ == "__main__":
    main()
