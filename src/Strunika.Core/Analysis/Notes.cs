namespace Strunika.Core.Analysis;

public static class Notes
{
    public static readonly string[] Names =
        { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

    public const double A4Frequency = 440.0;
    public const int A4Midi = 69;

    public static double MidiFromFrequency(double frequency) =>
        A4Midi + 12.0 * Math.Log2(frequency / A4Frequency);

    public static double FrequencyFromMidi(double midi) =>
        A4Frequency * Math.Pow(2.0, (midi - A4Midi) / 12.0);

    /// <summary>Nearest note name + octave and the deviation in cents.</summary>
    public static (string Name, int Octave, double Cents) Describe(double frequency)
    {
        double midi = MidiFromFrequency(frequency);
        int nearest = (int)Math.Round(midi);
        double cents = (midi - nearest) * 100.0;
        int pitchClass = ((nearest % 12) + 12) % 12;
        int octave = nearest / 12 - 1;
        return (Names[pitchClass], octave, cents);
    }

    public static int PitchClassFromFrequency(double frequency)
    {
        int nearest = (int)Math.Round(MidiFromFrequency(frequency));
        return ((nearest % 12) + 12) % 12;
    }
}
