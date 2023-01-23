using Microsoft.Extensions.Configuration;

namespace Services;

public static class Configuration
{
    public static string AiServiceUrl => Config.GetSection("AiServiceUrl").Value;
    public static string PlastimatchPath => Config.GetSection("PlastimatchPath").Value;
    public static string ModelOutputs => Config.GetSection("ModelsEvaluationDirectory").Value;
    public static string LogPath => Config.GetSection("LogPath").Value;
    public static bool EnableEvaluationSetup => bool.Parse(Config.GetSection("EnableEvaluationSetup").Value);

    private static readonly IConfigurationRoot Config = new ConfigurationBuilder()
        .AddJsonFile("appsettings.json")
        .Build();
}