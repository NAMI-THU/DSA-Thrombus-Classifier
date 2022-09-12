namespace Services.AiService.Requests;

public class LoadImagesRequest
{
    public string PathFrontal { get; set; }
    public string PathLateral { get; set; }
    public bool Normalized { get; set; }
}