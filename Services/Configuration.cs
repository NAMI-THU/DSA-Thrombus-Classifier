using System.Configuration;

namespace Services;

public static class Configuration
{
    public static string AiServiceUrl => ConfigurationManager.ConnectionStrings["AiServiceUrl"].ConnectionString;
    public static string PlastimatchPath => ConfigurationManager.AppSettings.Get("PlastimatchPath");
    public static string ModelOutputs => ConfigurationManager.AppSettings.Get("ModelsEvaluationDirectory");
}