
using System.Text.Json;

namespace Services.AiService.Interpreter;

public class ResultInterpreter
{
    // TODO: Perhaps distinguish between frontal and lateral for threshold ?
    private double _threshold;
    private List<Tuple<double, bool>> _testResultsValidationSet = new();
    private List<Tuple<double, bool>> _testResultsTestSet = new();
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
    
    public string TruePositivesPercentage { get; private set; } = "0%";

    public string TrueNegativesPercentage { get; private set; } = "0%";
    
    public string FalsePositivesPercentage { get; private set; } = "0%";
    
    public string FalseNegativesPercentage { get; private set; } = "0%";

    public string AccuracyString { get; private set; } = "0%";
    public string F1ScoreString { get; private set; } = "0%";
    public string PrecisionString { get; private set; } = "0%";
    public string RecallString { get; private set; } = "0%";


    public void LoadData()
    {
        // TODO: Make Async
        var outputs_validation = new List<FoldSingleResult>();
        var outputs_test = new List<FoldSingleResult>();
        for (var fold = 1; fold <= 5; fold++)
        {
            using FileStream stream = File.OpenRead(Path.Combine(Configuration.ModelOutputs,$"output_fold_{fold}.json"));
            using FileStream stream2 = File.OpenRead(Path.Combine(Configuration.ModelOutputs,$"output_fold_{fold}_test.json"));
            var foldResults = JsonSerializer.Deserialize<List<FoldSingleResult>>(stream);
            var foldResults2 = JsonSerializer.Deserialize<List<FoldSingleResult>>(stream2);
            outputs_validation.AddRange(foldResults ?? throw new InvalidOperationException("invalid data in output file"));
            outputs_test.AddRange(foldResults2 ?? throw new InvalidOperationException("invalid data in output file"));
        }

        _testResultsValidationSet.Clear();
        
        foreach (var fsr in outputs_validation)
        {
            var frontal = new Tuple<double, bool>(fsr.Frontal_Output, fsr.Frontal_Expected.Equals(1.0));
            var lateral = new Tuple<double, bool>(fsr.Lateral_Output, fsr.Lateral_Expected.Equals(1.0));
            _testResultsValidationSet.Add(frontal);
            _testResultsValidationSet.Add(lateral);
        }
        
        _testResultsTestSet.Clear();
        foreach (var fsr in outputs_test)
        {
            var frontal = new Tuple<double, bool>(fsr.Frontal_Output, fsr.Frontal_Expected.Equals(1.0));
            var lateral = new Tuple<double, bool>(fsr.Lateral_Output, fsr.Lateral_Expected.Equals(1.0));
            _testResultsTestSet.Add(frontal);
            _testResultsTestSet.Add(lateral);
        }
    }

    public bool HasThrombus(double rawModelOutput)
    {
        return HasThrombus(rawModelOutput, Threshold);
    }

    private bool HasThrombus(double rawModelOutput, double threshold)
    {
        return rawModelOutput >= threshold;
    }

    private (int, int, int, int) CalculateConfusionMatrixEntries(double threshold)
    {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        // Chose Test or Validation
        foreach (var tuple in _testResultsTestSet)
        {
            if (tuple.Item2)
            {
                if (HasThrombus(tuple.Item1, threshold))
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
                if (HasThrombus(tuple.Item1, threshold))
                {
                    fp++;
                }
                else
                {
                    tn++;
                }
            }
        }

        return (tp, tn, fp, fn);
    }
    private void CalculateConfusionMatrix()
    {
        var (tp, tn, fp, fn) = CalculateConfusionMatrixEntries(Threshold);

        TruePositives = tp;
        TrueNegatives = tn;
        FalseNegatives = fn;
        FalsePositives = fp;

        var sum = tp + tn + fn + fp;
        if (sum == 0)
        {
            TruePositivesPercentage = "0.00%";
            TrueNegativesPercentage = "0.00%";
            FalsePositivesPercentage = "0.00%";
            FalseNegativesPercentage = "0.00%";
            AccuracyString = "0%";
            RecallString = "0%";
            PrecisionString = "0%";
            F1ScoreString = "0%";
        }
        else
        {
            TruePositivesPercentage = $"{((double)tp / sum) * 100:F2}%";
            TrueNegativesPercentage = $"{((double)tn / sum) * 100:F2}%";
            FalseNegativesPercentage = $"{((double)fn / sum) * 100:F2}%";
            FalsePositivesPercentage = $"{((double)fp / sum) * 100:F2}%";
            
            var precision = (double)tp / (fp + tp);
            var recall = (double)tp / (fn + tp);
            if (double.IsNaN(precision)) { precision = 0;}
            if (double.IsNaN(recall)) { recall = 0;}
            
            AccuracyString = $"{((double)(tp +fp) / sum )* 100:F0}%";
            RecallString = $"{recall* 100:F0}%";
            PrecisionString = $"{precision* 100:F0}%";

            if (precision == 0 || recall == 0)
            {
                F1ScoreString = "0%";
            }
            else
            {
                F1ScoreString = $"{(2 * precision * recall) * 100 / (precision + recall):F0}%";
            }
        }
    }

    public double CalculateBestThreshold()
    {
        // We search threshold t=argmax sqrt((1-FPR)²+TPR²)
        // TODO: Look for better metric
        var best_t = 0.0;
        var distance = 0.0;
        for(var t = 0.0; t<=1.0;t+=0.005){
            var (tp, tn, fp, fn) = CalculateConfusionMatrixEntries(t);
            var tpr = (double) tp / (tp + fn);
            var fpr = (double) fp / (fp + tn);
            var d = Math.Sqrt(Math.Pow(1 - fpr, 2) + Math.Pow(tpr, 2));
            if (!(d >= distance)) continue;
            distance = d;
            best_t = t;
        }

        return best_t;
    }
}