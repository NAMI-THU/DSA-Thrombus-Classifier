using System.Diagnostics;
using CliWrap;

namespace Services.FileUtils;

public static class DicomConverter
{
    private static string _converterPath = Configuration.PlastimatchPath;
    
    public static async Task<string> Dicom2Nifti(string? inputPath)
    {
        // File checking and sanitizing:
        if (inputPath == null || !File.Exists(inputPath))
        {
            throw new ArgumentException("The specified file path was not valid.", nameof(inputPath));
        }
        
        // Check converter, TODO: Add security checks. Embed, check hash?
        if (!File.Exists(_converterPath))
        {
            var workingDirectory = Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName);
            var file = Path.Combine(workingDirectory, "plastimatch", "plastimatch.exe");
            if (File.Exists(file))
            {
                _converterPath = file;
            }
            else
            {
                throw new ArgumentException("Unable to find plastimatch. Please make sure it is in the exe's directory or specify the path in appsettings.json");
            }
        }
        
        var tmpFile = Path.GetTempFileName()+".nii";
        await Cli.Wrap(_converterPath)
            .WithArguments($"convert --input {inputPath} --output-img {tmpFile}")
            .ExecuteAsync();
        return tmpFile;
    }
}