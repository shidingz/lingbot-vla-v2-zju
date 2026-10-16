#!/usr/bin/env python3
"""Read-only audit of existing traces; output exclusively to a new audit JSON.
No torch import, accelerator initialization, subprocesses, or training changes.
"""
import argparse
import collections
import csv
import datetime
import json
import statistics
from pathlib import Path


def ns(text):
    """Parse epoch microseconds exactly as integer ns, avoiding float cancellation."""
    whole, _, frac = str(text).strip().partition('.')
    return int(whole) * 1000 + int((frac + '000')[:3])


def union_ns(intervals):
    end = None
    total = 0
    for left, right in sorted(intervals):
        total += right - left if end is None else max(0, right - max(left, end))
        end = right if end is None else max(end, right)
    return total


def inspect(output):
    with (output / 'step_trace_time.csv').open() as stream:
        official_rows = list(csv.DictReader(stream))
    count = len(official_rows)
    compute, communication = [], []
    counts = collections.Counter()
    durations = collections.Counter()
    core_counts = collections.Counter()
    first_hcom = None
    with (output / 'kernel_details.csv').open() as stream:
        reader = csv.reader(stream)
        headers = next(reader)
        indexes = {key: headers.index(key) for key in ['Name', 'Type', 'Accelerator Core', 'Start Time(us)', 'Duration(us)']}
        for line, row in enumerate(reader, 2):
            name, kind, core = (row[indexes[key]] for key in ['Name', 'Type', 'Accelerator Core'])
            start, duration = (ns(row[indexes[key]]) for key in ['Start Time(us)', 'Duration(us)'])
            is_communication = core.upper() in {'HCCL', 'COMMUNICATION'} or 'hccl' in name.lower() or name.lower().startswith('hcom') or kind.lower().startswith('hcom')
            (communication if is_communication else compute).append((start, start + duration))
            core_counts[core] += 1
            if is_communication:
                counts[kind] += 1
                durations[kind] += duration
                if first_hcom is None:
                    first_hcom = {'line': line, 'Name': name, 'Type': kind, 'Accelerator Core': core}
    scale = count * 1e9
    compute_s, communication_s, combined_s = (union_ns(v) / scale for v in [compute, communication, compute + communication])
    official = {key: statistics.mean(float(row[key]) for row in official_rows) / 1e6 for key in official_rows[0] if key not in {'Step', 'Device_id'}}
    comm_json = json.loads((output / 'communication.json').read_text())
    breakdown = collections.Counter()
    links = collections.defaultdict(collections.Counter)
    zero_breakdown = comm_ops = 0
    for step in comm_json.values():
        for operators in step.values():
            for name, values in operators.items():
                if name.lower().startswith('total'):
                    continue
                info = values.get('Communication Time Info', {})
                comm_ops += 1
                zero_breakdown += all(info.get(key, 0) == 0 for key in ['Transit Time(ms)', 'Wait Time(ms)', 'Synchronization Time(ms)'])
                breakdown.update({key: value for key, value in info.items() if key.endswith('(ms)')})
                for link, value in values.get('Communication Bandwidth Info', {}).items():
                    links[link]['MB'] += value.get('Transit Size(MB)', 0)
                    links[link]['ms'] += value.get('Transit Time(ms)', 0)
    return {
        'rank': output.parents[1].name,
        'source_directory': str(output),
        'recorded_steps': count,
        'profiler_step_ids': [row['Step'] for row in official_rows],
        'raw_compute_rows': len(compute),
        'raw_communication_rows': len(communication),
        'core_counts': dict(core_counts),
        'first_hcom_evidence': first_hcom,
        'corrected_union_s_per_step': {'compute': compute_s, 'communication': communication_s, 'overlapped': compute_s + communication_s - combined_s, 'communication_not_overlapped': combined_s - compute_s, 'compute_or_communication': combined_s},
        'official_step_trace_s_per_step': official,
        'corrected_minus_official_s': {'compute': compute_s - official['Computing'], 'communication': communication_s - official['Communication'], 'communication_not_overlapped': combined_s - compute_s - official['Communication(Not Overlapped)']},
        'collective_calls_per_step': {key: value / count for key, value in counts.items()},
        'collective_duration_sum_s_per_step': {key: value / scale for key, value in durations.items()},
        'communication_time_info': {'ops': comm_ops, 'all_transit_wait_sync_zero_ops': zero_breakdown, 'sums_s_per_step': {key: value / count / 1000 for key, value in breakdown.items()}},
        'communication_bandwidth_info': {key: {'transit_MB_per_step': value['MB'] / count, 'transit_sum_s_per_step': value['ms'] / count / 1000, 'aggregate_GB_s': value['MB'] / value['ms'] if value['ms'] else 0} for key, value in links.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-root', type=Path, default=Path('/cache/lingbot/profiling/robotwin_20261016_1530/trace'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('communication_audit.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite existing audit output')
    ranks = []
    for path in sorted(args.trace_root.glob('rank*/*/ASCEND_PROFILER_OUTPUT')):
        rank = inspect(path)
        ranks.append(rank)
        print(rank['rank'], json.dumps(rank['corrected_union_s_per_step']), flush=True)
    if len(ranks) != 8 or any(rank['recorded_steps'] != 4 for rank in ranks):
        raise SystemExit('Unexpected capture coverage')
    output = {
        'audited_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'scope': 'Existing 2026-10-16 15:30 isolated profiler capture only; no new training or accelerator operations.',
        'algorithm': 'Classify core COMMUNICATION/HCCL and names/types hcom or hccl as communication. Parse timestamp and duration microseconds to integer nanoseconds, merge intervals per rank, divide by four captured steps. Overlap = compute_union + communication_union - combined_union. Exclusive communication = combined_union - compute_union. No sums across devices are interpreted as wall time.',
        'existing_script_bug': {'file': 'tools/analyze_npu_profile.py', 'line': 65, 'existing_rule': "core == 'HCCL' or 'hccl' in Name.lower()", 'actual_schema': "core='COMMUNICATION', Name='hcom_allGather__...'", 'impact': 'All communication was counted as compute in auxiliary union fields; communication_union_s_per_step=0 was incorrect. Existing findings based on official step_trace_time.csv remain valid.'},
        'mean_corrected_union_s_per_step': {key: statistics.mean(rank['corrected_union_s_per_step'][key] for rank in ranks) for key in ranks[0]['corrected_union_s_per_step']},
        'mean_official_step_trace_s_per_step': {key: statistics.mean(rank['official_step_trace_s_per_step'][key] for rank in ranks) for key in ranks[0]['official_step_trace_s_per_step']},
        'semantics_and_limits': [
            'Computing includes intervals overlapping communication. Communication(Not Overlapped) is exclusive of compute; Free is neither recorded compute nor communication. These are device timeline categories, not host-cause attribution or arithmetic units utilization.',
            'The capture has about 30.17 percent overhead and only four steps per rank. Do not extrapolate absolute profiled times to the complete epoch.',
            'All 1016 operations per rank have zero Transit/Wait/Synchronization in Communication Time Info and Idle equal to elapsed. Positive HCCS bandwidth subrecords show actual transfer. This breakdown cannot establish zero wait, zero transfer, all idle, or a network bandwidth bottleneck.',
            'Installed CANN parser selects master-plane events for time analysis and all planes for bandwidth analysis; master-plane subtask coverage/semantics needs further inspection to explain the zero breakdown.',
            'SDMA is an aggregate including HCCS, PCIe and SIO. HCCS and SDMA quantities must not be added. Bandwidth subtask sums are not elapsed critical-path transmission time or peak physical-link utilization.',
            'The old float-epoch union implementation adds submillisecond rounding error; integer nanoseconds avoid that error. Official step CSV and exact raw union may still differ from range selection or source representation.'
        ],
        'definition_sources': [
            'https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0038.html',
            'https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0066.html',
            '/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/torch_npu/profiler/analysis/prof_view/_trace_step_time_parser.py:151',
            '/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msparser/cluster/communication_parser.py:175',
            '/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msparser/cluster/communication_parser.py:220',
            '/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msparser/cluster/meta_parser.py:157'
        ],
        'ranks': ranks,
    }
    with args.output.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('WROTE', args.output, flush=True)


if __name__ == '__main__':
    main()
