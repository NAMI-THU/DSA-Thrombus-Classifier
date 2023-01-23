using System.IO;
using System.Linq;

namespace UI.Util;

public static class Validation
{
    public static bool CheckModelFolderStructure(string path)
    {
        var fPath = Path.Combine(path, "frontal");
        var lPath = Path.Combine(path, "lateral");
        if (Directory.Exists(fPath) && Directory.Exists(lPath))
        {
            var fFiles = Directory.GetFiles(fPath);
            var lFiles = Directory.GetFiles(lPath);

            if (fFiles.Any(f => f.EndsWith(".pt")) && lFiles.Any(f => f.EndsWith(".pt")))
            {
                if (fFiles.Where(f => f.EndsWith(".pt")).Select(Path.GetFileName).All(f => lFiles.Select(Path.GetFileName).Contains(f)))
                {
                    return true;
                }
            }
        }

        return false;
    }
}