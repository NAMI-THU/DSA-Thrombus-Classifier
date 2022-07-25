using System.Runtime.InteropServices;
using System.Text;
using Newtonsoft.Json;
using Services.AiService.Requests;
using Services.AiService.Responses;

namespace Services.AiService;

public class AiServiceCommunication
{
    // TODO: Change
    private static readonly HttpClient Client = new() {BaseAddress = new Uri("http://127.0.0.1:5000/") };

    
    public static async Task<ClassificationResponse> ClassifySequence(string pathToFrontal, string pathToLateral)
    {
        var request = new ClassificationRequest { PathFrontal = pathToFrontal, PathLateral = pathToLateral };
        var response = await Post(request, "/AiService/Classification");

        var content = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<ClassificationResponse>(content) ?? throw new ExternalException("Failed to convert servers response");
    }
    
    private static async Task<HttpResponseMessage> Post(object request, string uri) {
        var json = JsonConvert.SerializeObject(request);
        var data = new StringContent(json, Encoding.UTF8, "application/json");
            
        var response = await Client.PostAsync(uri, data);
        response.EnsureSuccessStatusCode();

        return response;
    }
}