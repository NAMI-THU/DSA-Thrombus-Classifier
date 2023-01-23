namespace Services.AiService.Requests;

public class ClassificationRequest
{
    public string? PathFrontal { get; set; }
    public string? PathLateral { get; set; }
    public string? ModelFrontal { get; set; }
    public string? ModelLateral { get; set; }
}