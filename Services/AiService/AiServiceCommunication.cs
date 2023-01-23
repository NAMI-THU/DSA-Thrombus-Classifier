using System.Runtime.InteropServices;
using System.Text;
using Newtonsoft.Json;
using Services.AiService.Requests;
using Services.AiService.Responses;

namespace Services.AiService;

public static class AiServiceCommunication
{
    private static readonly HttpClient Client = new() {BaseAddress = new Uri($"http://{Configuration.AiServiceUrl}/"), Timeout = TimeSpan.FromMinutes(5)};
    
    public static async Task<ClassificationResponse> ClassifySequence(string modelFrontal, string modelLateral, string fileFrontal, string fileLateral)
    {
        var request = new ClassificationRequest { PathFrontal = fileFrontal, PathLateral = fileLateral, ModelFrontal = modelFrontal, ModelLateral = modelLateral };
        var response = await Post(request, "/AiService/Classification");

        var content = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<ClassificationResponse>(content) ?? throw new ExternalException("Failed to convert servers response");
    }

    public static async Task PreloadModels(string directory)
    {
        var request = new LoadModelsRequest { Directory = directory};
        await Post(request, "/AiService/PreloadModels");
    }

    public static async Task<ImageResponse> LoadImages(string imageFrontal, string imageLateral, bool normalized=false)
    {
        var request = new LoadImagesRequest
            { PathFrontal = imageFrontal, PathLateral = imageLateral, Normalized = normalized };
        var response = await Post(request, "/AiService/LoadImages");
        var content = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<ImageResponse>(content) ??
               throw new ExternalException("Failed to convert image response");
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