import os
import subprocess
import tempfile
from tqdm import tqdm

# Directory containing the copied video files
videos_dir = r"C:\Users\athar\Documents\GitHub\testcv\videos_fake"

# Count for statistics
total_files_converted = 0
failed_conversions = 0

# Check if ffmpeg is available
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print("FFmpeg is available. Starting conversion...")
except (subprocess.SubprocessError, FileNotFoundError):
    print("ERROR: FFmpeg is not installed or not in PATH. Please install FFmpeg first.")
    exit(1)

# Get all files in the directory
all_files = os.listdir(videos_dir)
video_files = [f for f in all_files if os.path.isfile(os.path.join(videos_dir, f)) and not f.lower().endswith('.mov')]
print(f"Found {len(video_files)} files to convert")

# Process files with progress bar
for filename in tqdm(video_files, desc="Converting videos to MOV", unit="file"):
    input_path = os.path.join(videos_dir, filename)
    
    # Create output path with .MOV extension
    output_path = os.path.splitext(input_path)[0] + ".MOV"
    
    # Create a temporary file for conversion
    temp_output = tempfile.NamedTemporaryFile(suffix='.MOV', delete=False).name
    
    try:
        # Run FFmpeg conversion (quiet mode for cleaner output with tqdm)
        result = subprocess.run([
            "ffmpeg",
            "-i", input_path,
            "-c:v", "prores_ks",  # ProRes codec (common for .MOV)
            "-profile:v", "3",     # ProRes 422 HQ profile
            "-c:a", "pcm_s16le",   # Uncompressed audio
            "-y",                  # Overwrite output files
            "-loglevel", "error",  # Reduce FFmpeg output for cleaner tqdm display
            temp_output
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            # Remove original file
            os.remove(input_path)
            
            # Rename temp file to final output path
            os.rename(temp_output, output_path)
            
            total_files_converted += 1
        else:
            failed_conversions += 1
            tqdm.write(f"Failed to convert {filename}. Error: {result.stderr.decode()}")
            # Clean up temp file if conversion failed
            if os.path.exists(temp_output):
                os.remove(temp_output)
    
    except Exception as e:
        failed_conversions += 1
        tqdm.write(f"Error processing {filename}: {str(e)}")
        # Clean up temp file if there was an exception
        if os.path.exists(temp_output):
            os.remove(temp_output)

print("\nConversion Summary:")
print(f"Total files converted to .MOV: {total_files_converted}")
print(f"Failed conversions: {failed_conversions}")