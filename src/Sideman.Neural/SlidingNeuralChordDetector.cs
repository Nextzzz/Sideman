namespace Sideman.Neural;

/// <summary>
/// Live BTC inference over a sliding 10-second window — the "confirmed"
/// tier of the two-tier display. Feed 44100 Hz mic chunks continuously,
/// call <see cref="Tick"/> periodically (e.g. every 250 ms) from any
/// background thread; the confirmed chord lands ~1 s behind real time.
///
/// Design: CQT frames are cached and only NEW frames are computed per
/// tick (~5-6), so a tick costs ~50-80 ms instead of recomputing the
/// whole window (~1 s). The reported frame sits a few frames behind the
/// newest one so it has bidirectional context inside the window.
/// </summary>
public sealed class SlidingNeuralChordDetector : IDisposable
{
    private const int Hop = CqtExtractor.Hop;
    private const int WindowFrames = 108;
    private const int FutureMarginFrames = 4; // in-window future context

    private readonly NeuralChordRecognizer _recognizer;
    private readonly CqtExtractor _cqt = new();
    private readonly HalfbandDecimator _decimator = new();

    // 12 seconds of 22050 Hz audio, linearized per tick for frame math.
    private readonly float[] _ring = new float[CqtExtractor.SampleRate * 12];
    private long _written;
    private readonly object _audioLock = new();

    private readonly List<float[]> _frames = new(); // rolling CQT frame cache
    private long _firstCachedFrame;                 // global index of _frames[0]
    private readonly float[] _linear;
    private int _busy;

    public string CurrentLabel { get; private set; } = "N";

    /// <summary>Display-ready confirmed chord ("—", "C", "F#m7"...).</summary>
    public string CurrentPretty => ChordLabels.Pretty(CurrentLabel);

    /// <summary>A new label must win this many consecutive inferences
    /// before being confirmed. During a chord change the window briefly
    /// contains both chords and a single inference can land anywhere —
    /// one-tick blips must never reach the display or history.</summary>
    public int ConfirmTicks { get; set; } = 2;

    private string _pendingLabel = "N";
    private int _pendingCount;

    public event Action<string>? ConfirmedChanged;

    public SlidingNeuralChordDetector(string onnxPath)
    {
        _recognizer = new NeuralChordRecognizer(onnxPath);
        _linear = new float[_ring.Length];
    }

    /// <summary>Feed microphone samples at 44100 Hz (any chunk size).</summary>
    public void AddSamples(ReadOnlySpan<float> chunk44100)
    {
        var downsampled = _decimator.Process(chunk44100);
        lock (_audioLock)
        {
            foreach (var s in downsampled)
                _ring[(int)(_written++ % _ring.Length)] = s;
        }
    }

    /// <summary>
    /// Compute newly available CQT frames and, when anything new arrived,
    /// run inference. Safe to call from a timer; overlapping calls are
    /// skipped. Returns true if the confirmed chord changed.
    /// </summary>
    public bool Tick()
    {
        if (Interlocked.Exchange(ref _busy, 1) == 1)
            return false;
        try
        {
            long written;
            lock (_audioLock)
            {
                written = _written;
                // Linearize the ring: _linear[i] = sample (written - length + i).
                for (int i = 0; i < _ring.Length; i++)
                {
                    long global = written - _ring.Length + i;
                    _linear[i] = global < 0 ? 0f : _ring[(int)(global % _ring.Length)];
                }
            }

            // A frame is computable once its full kernel context has arrived.
            long newestComputable = (written - _cqt.MaxKernelHalf) / Hop;
            if (newestComputable < 0)
                return false;

            bool added = false;
            long next = _firstCachedFrame + _frames.Count;
            for (; next <= newestComputable; next++)
            {
                long centerGlobal = next * Hop;
                int centerLocal = (int)(centerGlobal - (written - _ring.Length));
                if (centerLocal < 0)
                    continue; // scrolled out before we got to it
                var row = new float[CqtExtractor.Bins];
                _cqt.ComputeFrame(_linear, centerLocal, row);
                if (_frames.Count == 0)
                    _firstCachedFrame = next;
                _frames.Add(row);
                added = true;
            }
            if (!added)
                return false;

            // Trim the cache to one window.
            while (_frames.Count > WindowFrames)
            {
                _frames.RemoveAt(0);
                _firstCachedFrame++;
            }

            if (_frames.Count <= FutureMarginFrames)
                return false;

            var labels = _recognizer.PredictWindow(_frames);
            string label = labels[Math.Max(0, labels.Length - 1 - FutureMarginFrames)];
            if (label == CurrentLabel)
            {
                _pendingCount = 0;
                return false;
            }

            // Stability gate: the same NEW label must repeat across ticks.
            if (label == _pendingLabel)
            {
                _pendingCount++;
            }
            else
            {
                _pendingLabel = label;
                _pendingCount = 1;
            }
            if (_pendingCount < ConfirmTicks)
                return false;

            CurrentLabel = label;
            _pendingCount = 0;
            ConfirmedChanged?.Invoke(label);
            return true;
        }
        finally
        {
            Interlocked.Exchange(ref _busy, 0);
        }
    }

    public void Dispose() => _recognizer.Dispose();
}
