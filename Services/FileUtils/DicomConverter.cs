using System.Configuration;
using CliWrap;

namespace Services.FileUtils;

public class DicomConverter
{
    private static readonly string ConverterPath = Configuration.PlastimatchPath;
    
    public static async Task<string> Dicom2Nifti(string? inputPath)
    {
        // File checking and sanitizing:
        if (inputPath == null || !File.Exists(inputPath))
        {
            throw new ArgumentException("The specified file path was not valid.", nameof(inputPath));
        }
        
        // Check converter, TODO: Add security checks. Embed, check hash?
        if (!File.Exists(ConverterPath))
        {
            throw new ArgumentException("Unable to find converter.");
        }
        
        var tmpFile = Path.GetTempFileName()+".nii";
        await Cli.Wrap(ConverterPath)
            .WithArguments($"convert --input {inputPath} --output-img {tmpFile}")
            .ExecuteAsync();
        return tmpFile;
    }
}