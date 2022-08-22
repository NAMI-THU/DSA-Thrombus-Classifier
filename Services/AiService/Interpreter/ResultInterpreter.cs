namespace Services.AiService.Interpreter;

public class ResultInterpreter
{
    private double _threshold;
    private List<Tuple<double, bool>> _testResults = new();
    public double Threshold
    {
        get => _threshold;
        set
        {
            _threshold = value;
            CalculateConfusionMatrix();
        }
    }

    public int TruePositives { get; private set; }

    public int TrueNegatives { get; private set; }
    
    public int FalsePositives { get; private set; }
    
    public int FalseNegatives { get; private set; }


    public ResultInterpreter()
    {
        // Load Raw outputs together with labels
        // TODO _testResults =
    }

    public bool HasThrombus(double rawModelOutput)
    {
        return rawModelOutput >= Threshold;
    }

    private void CalculateConfusionMatrix()
    {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        foreach (var tuple in _testResults)
        {
            if (tuple.Item2)
            {
                if (HasThrombus(tuple.Item1))
                {
                    tp++;
                }
                else
                {
                    fn++;
                }
            }
            else
            {
                if (HasThrombus(tuple.Item1))
                {
                    fp++;
                }
                else
                {
                    tn++;
                }
            }
        }

        TruePositives = tp;
        TrueNegatives = tn;
        FalseNegatives = fn;
        FalsePositives = fp;
    }
}