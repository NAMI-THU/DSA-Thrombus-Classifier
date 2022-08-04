using System.Configuration;
using System.Runtime.InteropServices;
using System.Text;
using Newtonsoft.Json;
using Services.AiService.Requests;
using Services.AiService.Responses;

namespace Services.AiService;

public static class AiServiceCommunication
{
    // TODO: Quite a security issue here!
    private static readonly HttpClient Client = new() {BaseAddress = new Uri($"http://{Configuration.AiServiceUrl}/"), Timeout = TimeSpan.FromMinutes(5)};

    
    public static async Task<ClassificationResponse> ClassifySequence(string pathToFrontal, string pathToLateral)
    {
        var request = new ClassificationRequest { PathFrontal = pathToFrontal, PathLateral = pathToLateral };
        var response = await Post(request, "/AiService/Classification");

        var content = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<ClassificationResponse>(content) ?? throw new ExternalException("Failed to convert servers response");
    }

    public static async Task PreloadModels()
    {
        await Get("/AiService/PreloadModels");
    }
    
    private static async Task<HttpResponseMessage> Post(object request, string uri) {
        var json = JsonConvert.SerializeObject(request);
        var data = new StringContent(json, Encoding.UTF8, "application/json");
            
        var response = await Client.PostAsync(uri, data);
        response.EnsureSuccessStatusCode();

        return response;
    }

    private static async Task<HttpResponseMessage> Get( string uri)
    {
        var response = await Client.GetAsync(uri);
        response.EnsureSuccessStatusCode();
        return response;
    }
}