namespace UI.Util;

public record Attribution(string Library, License License, string Author);

public class License
{
    public License(string name, string url)
    {
        Name = name;
        Url = url;
    }

    public string Name { get; }
    public string Url { get; }


    public override string ToString()
    {
        return Name;
    }
}