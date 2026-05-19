import mne, numpy as np, pathlib, glob
# Find the folder with glob
matches = glob.glob(r'C:\Users\hasla\Desktop\old\bj*rn_04_06_25\P01\data\task_stim.vhdr')
if not matches:
    import os
    for d in os.listdir(r'C:\Users\hasla\Desktop\old'):
        if d.startswith('bj') and '04_06_25' in d:
            matches = [os.path.join(r'C:\Users\hasla\Desktop\old', d, 'P01', 'data', 'task_stim.vhdr')]
            break
print('Using:', matches[0])
raw = mne.io.read_raw_brainvision(matches[0], preload=True)
events, event_id = mne.events_from_annotations(raw)
sfreq = raw.info['sfreq']
print('=== FIRST 30 EVENTS ===')
for i in range(min(30, len(events))):
    s, z, c = events[i]
    print(f'  {i:3d}: sample={s:8d}  code={c:3d}  time={s/sfreq:.4f}s')
print()
print('=== EVENT CODE COUNTS ===')
codes, counts = np.unique(events[:,2], return_counts=True)
for co, cn in zip(codes, counts):
    label = [k for k,v in event_id.items() if v==co]
    lbl = label[0] if label else '?'
    print(f'  code {co:3d} ({lbl:>20s}): {cn}')
print()
print('=== INTER-EVENT INTERVALS (first 30) ===')
for i in range(1, min(30, len(events))):
    dt = (events[i,0] - events[i-1,0]) / sfreq
    print(f'  {i-1}->{i}: {dt:.4f}s')
print()
print('=== FIRST 100 EVENT CODES ===')
print(list(events[:100,2]))
print()
print('EVENT_ID:', event_id)
print('TOTAL:', len(events))
