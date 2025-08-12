import requests
import json
import urllib.parse
import time
import os
import subprocess
import pymongo
from faster_whisper import WhisperModel, BatchedInferencePipeline
import librosa
import soundfile as sf
import torch
import torchaudio.transforms as T
from snac import SNAC

MONGO_URI = "mongodb://root:9AsYmXYKmYLHcNsShmCb3L5DZMXH77rQ9GBRxm0HKownNWLwdzH9dW7zhPG9mpuR@46.4.101.229:8281/?directConnection=true"
COLLECTION_NAME = "tts_data"

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
client = pymongo.MongoClient(MONGO_URI)
db = client["tts_data"]
collection = db[COLLECTION_NAME]

model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2")
batched_model = BatchedInferencePipeline(model)

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to(device)


class ApiService:
    def __init__(self):
        self.client = requests.Session()
        self.auth_cookie = None
        self.kb_domain = "www.kb.dk"
        self.api_domain = "api.kaltura.nordu.net"
        self.ds_api_domain = "www.kb.dk"
        self.kaltura_partner_id = "397"
        self.kaltura_widget_id = "_397"
        self.kaltura_player_version = "html5:v3.14.4"

    def fetch_data(self, url):
        """Henter rå tekstdata fra en given URL."""
        headers = {'User-Agent': 'Mozilla/5.0'}

        if self.auth_cookie:
            headers['Cookie'] = self.auth_cookie

        try:
            response = self.client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Kunne ikke hente data fra {url}: {e}")
            return None

    def _generate_kaltura_stream_link(self, entry_id: str, flavor_id: str, file_ext: str) -> str:
        """
        Genererer et komplet Kaltura stream-link ud fra entryId, flavorId og filendelse.
        """
        return (
            f"https://vod-cache.kaltura.nordu.net/p/{self.kaltura_partner_id}/sp/{self.kaltura_partner_id}00/serveFlavor/"
            f"entryId/{entry_id}/v/12/flavorId/{flavor_id}/name/a.{file_ext}"
        )

    def extract_media_url_from_kaltura_response(self, response_data):
        """
        Udtrækker media URL. Bruger nu altid _generate_kaltura_stream_link for at få en direkte MP4 flavor URL.
        Forventer et multirequest-svar fra Kaltura.
        """
        try:
            data = json.loads(response_data)
            # context_object = data[2] # Not strictly needed if we don't use flavor_assets directly from here for HLS
            # flavor_assets = context_object.get('flavorAssets', []) # Not strictly needed
            sources = data[2].get('sources', []) # Still need sources to get a flavorId

            # We need an entry_id and a flavor_id to build the serveFlavor URL.
            # file_ext will be determined by the flavor if possible, or default.
            
            media_object_list = data[1].get('objects', [])
            if not media_object_list:
                print("Manglende 'objects' i Kaltura-respons data[1].")
                return None
            
            media_object = media_object_list[0]
            entry_id = media_object.get('id', '')

            current_flavor_id = None
            file_ext = "mp4" # Default to mp4, can be overridden if flavor asset info is available

            # Try to get flavorId from sources if available
            if isinstance(sources, list) and sources:
                 # Assuming the first source's flavorId is relevant for a downloadable MP4
                 # The 'sources' array often contains multiple formats and qualities.
                 # We need to pick one that is likely to be a simple video file.
                 # Let's iterate to find one with 'video/mp4' or a common video format
                 found_flavor_for_mp4 = False
                 for source_item in sources:
                     if isinstance(source_item, dict):
                         s_format = source_item.get('format')
                         s_mimetype = source_item.get('mimetype')
                         # Prioritize a flavorId that seems to be for an MP4
                         if s_mimetype == 'video/mp4' or s_format == 'url': # 'url' format sometimes links to MP4
                             temp_flavor_id = source_item.get('flavorIds')
                             if temp_flavor_id: # flavorIds can be a string like "0_xxxx,0_yyyy"
                                 current_flavor_id = temp_flavor_id.split(',')[0] # Take the first one
                                 # Check if flavorAssets has more info on this flavorId
                                 flavor_assets = data[2].get('flavorAssets', [])
                                 if isinstance(flavor_assets, list):
                                     for asset in flavor_assets:
                                         if asset.get('id') == current_flavor_id and asset.get('fileExt'):
                                             file_ext = asset.get('fileExt')
                                             break
                                 found_flavor_for_mp4 = True
                                 break
                 if not found_flavor_for_mp4 and isinstance(sources, list) and sources: # Fallback to first if no explicit mp4 found
                     current_flavor_id = sources[0].get('flavorIds','').split(',')[0]


            # If flavorId is still not found, try getting it from flavorAssets as a last resort
            # This part of logic might be less reliable as flavorAssets might not directly map
            # to a simple downloadable flavor if sources didn't provide one.
            if not current_flavor_id:
                flavor_assets = data[2].get('flavorAssets', [])
                if isinstance(flavor_assets, list) and flavor_assets:
                    # Heuristic: pick the first flavor asset that is not 'audio*' or 'image*' if possible
                    # and hope it's a video.
                    for asset in flavor_assets:
                        tags = asset.get('tags', '')
                        if 'audio' not in tags and 'image' not in tags and 'caption' not in tags: # try to avoid non-video
                            current_flavor_id = asset.get('id')
                            file_ext = asset.get('fileExt', 'mp4')
                            break
                    if not current_flavor_id and flavor_assets: # If still nothing, just take the first one
                         current_flavor_id = flavor_assets[0].get('id')
                         file_ext = flavor_assets[0].get('fileExt', 'mp4')


            if not (entry_id and current_flavor_id):
                print(f"Manglende data til at bygge media URL (entry_id: {entry_id}, flavor_id: {current_flavor_id}).")
                # Print more context if URL generation fails
                print(f"  entry_id from data[1]: {entry_id}")
                print(f"  Attempted current_flavor_id: {current_flavor_id}")
                print(f"  Sources object: {str(sources)[:200]}...")
                print(f"  FlavorAssets object: {str(data[2].get('flavorAssets', []))[:200]}...")
                return None

            # Ensure file_ext is sensible
            if not file_ext or len(file_ext) > 5: # basic sanity check
                file_ext = "mp4"

            print(f"  Generating serveFlavor URL with entry_id: {entry_id}, flavor_id: {current_flavor_id}, ext: {file_ext}")
            media_url = self._generate_kaltura_stream_link(entry_id, current_flavor_id, file_ext)
            return media_url

        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            print(f"Kunne ikke parse media-url fra Kaltura-respons: {e}")
            print(f"Response data snippet: {str(response_data)[:500]}")
            return None
        except Exception as e:
            print(f"Uventet fejl under parsing af Kaltura-respons: {e}")
            return None

    def fetch_kaltura_data(self, entry_id):
        """Henter metadata og afspilningsinformation for en specifik Kaltura entry."""
        url = f"https://{self.api_domain}/api_v3/service/multirequest"
        json_payload = {
            "1": {
                "service": "session",
                "action": "startWidgetSession",
                "widgetId": self.kaltura_widget_id
            },
            "2": {
                "service": "baseEntry",
                "action": "list",
                "ks": "{1:result:ks}",
                "filter": {"redirectFromEntryId": entry_id},
                "responseProfile": {
                    "type": 1,
                    "fields": "id,referenceId,name,duration,description,thumbnailUrl,dataUrl,duration,msDuration,flavorParamsIds,mediaType,type,tags,startTime,date,dvrStatus,externalSourceType,status"
                }
            },
            "3": {
                "service": "baseEntry",
                "action": "getPlaybackContext",
                "entryId": "{2:result:objects:0:id}",
                "ks": "{1:result:ks}",
                "contextDataParams": {
                    "objectType": "KalturaContextDataParams",
                    "flavorTags": "all"
                }
            },
            "4": {
                "service": "metadata_metadata",
                "action": "list",
                "filter": {
                    "objectType": "KalturaMetadataFilter",
                    "objectIdEqual": "{2:result:objects:0:id}",
                    "metadataObjectTypeEqual": "1"
                },
                "ks": "{1:result:ks}"
            },
            "apiVersion": "3.3.0",
            "format": 1,
            "ks": "",
            "clientTag": self.kaltura_player_version,
            "partnerId": self.kaltura_partner_id
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Host': self.api_domain,
            'Referer': f'https://{self.kb_domain}/find-materiale/dr-arkivet/',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0',
            'Content-Type': 'application/json'
        }

        if self.auth_cookie:
            headers['Cookie'] = f"Authorization={self.auth_cookie}"

        try:
            response = self.client.post(url, json=json_payload, headers=headers)
            response.raise_for_status()
            # logging.debug(f"Kaltura response for entry {entry_id}: {response.text}")
            return response.text
        except requests.RequestException as e:
            print(f"Kunne ikke hente Kaltura-data for entry {entry_id}: {e}")
            return None

    def authenticate(self, on_complete):
        """
        Udfører autentifikation mod KB-API'en og gemmer auth-cookie til senere brug.
        'on_complete' er en callback-funktion, der kaldes uanset resultat.
        """
        current_unix_time = int(time.time())

        cookie_header = (
            f"""ppms_privacy_6c58358e-1595-4533-8cf8-9b1c061871d0={{"visitorId":"0478c604-ce60-4537-8e17-fdb53fcd5c31","domain":{{"normalized":"{self.kb_domain}","isWildcard":false,"pattern":"{self.kb_domain}"}},"consents":{{"analytics":{{"status":1}}}}}}; """
            f"""CookieScriptConsent={{"bannershown":1,"action":"reject","consenttime":{current_unix_time},"categories":"[]","key":"99a8bf43-ba89-444c-9333-2971c53e72a6"}}"""
        )

        auth_url = f"https://{self.ds_api_domain}/ds-api/bff/v1/authenticate/"
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Cookie': cookie_header,
            'Referer': f'https://{self.kb_domain}/find-materiale/dr-arkivet/'
        }

        try:
            response = self.client.get(auth_url, headers=headers)
            response.raise_for_status()
            cookies = response.cookies.get_dict()
            auth_cookie = cookies.get("Authorization")
            if auth_cookie:
                self.auth_cookie = auth_cookie
                print("Autentificering gennemført og auth-cookie gemt.")
            else:
                print("Ingen Authorization-cookie fundet i svaret.")
        except requests.RequestException as e:
            print(f"Autentificering mislykkedes: {e}")
        finally:
            on_complete()  

    def fetch_search_results(self, search_term="*:*", start_index=0, sort_option="startTime asc", rows=10, media_type="", year_start=2005, year_end=2026, month_number=1):
        """
        Henter søgeresultater fra KB's DR-arkiv-API.
        Understøtter medietype-filtrering for 'ds.radio' og 'ds.tv'.
        """
        encoded = urllib.parse.quote(search_term, safe='*')
        media_filter = self._build_media_filter(media_type)

        url = (
            f"https://{self.ds_api_domain}/ds-api/bff/v1/proxy/search/?q={encoded}{media_filter}"
            f"&facet=false&start={start_index}&sort={urllib.parse.quote(sort_option)}&rows={rows}"
            f"&fq=startTime:[{year_start}-12-31T23:00:00.000Z TO {year_end}-12-31T22:59:59.999Z]"
            f"&fq=temporal_start_month:{month_number}"
        )

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Host': self.ds_api_domain,
            'Referer': f'https://{self.kb_domain}/find-materiale/dr-arkivet/find',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0'
        }

        if self.auth_cookie:
            headers['Cookie'] = f"Authorization={self.auth_cookie}"

        try:
            response = self.client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            print(f"HTTP {response.status_code} ved forespørgsel til søge-API: {e}")
            return None
        except requests.RequestException as e:
            print(f"Forespørgsel til søge-API mislykkedes: {e}")
            return None
        except json.JSONDecodeError:
            print("Kunne ikke parse JSON-respons fra søge-API.")
            return None

    def _build_media_filter(self, media_type):
        """Bygger media filter strengen baseret på media type."""
        if media_type in ("ds.radio", "ds.tv"):
            return f"&fq=origin%3A%22{media_type}%22"
        return ""

    def parse_search_response(self, response_data):
        """
        Parser JSON-streng til Python-objekt.
        Returnerer None hvis input er ugyldigt.
        """
        try:
            return json.loads(response_data) if response_data else None
        except json.JSONDecodeError as e:
            print(f"Kunne ikke parse søge-respons: {e}")
            return None

    def download_media(self, media_url, filename, download_path="video_files"):
        """Downloader medie fra en URL og gemmer det i den specificerede sti."""
        if not media_url:
            print("  Download skipped: No media URL provided.")
            return None # Return None to indicate failure/skip

        try:
            if not os.path.exists(download_path):
                os.makedirs(download_path)
            
            filepath = os.path.join(download_path, filename)

            print(f"  Downloading {media_url} to {filepath}...")
            response = self.client.get(media_url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Successfully downloaded {filepath}")
            return filepath # Return the path to the downloaded file
        except requests.RequestException as e:
            print(f"  Failed to download {media_url}: {e}")
            return None
        except IOError as e:
            print(f"  Failed to save file {filepath}: {e}")
            return None
        except Exception as e:
            print(f"  An unexpected error occurred during download: {e}")
            return None

    def extract_audio(self, input_filepath, output_filename, output_path="audio_files"):
        """Extract audio from a local media file using PyAV.
           Saves the audio as an MP3 file.
        """
        if not input_filepath or not os.path.exists(input_filepath):
            print(f"  Audio extraction skipped: Input file not provided or does not exist: {input_filepath}")
            return False

        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            base, ext = os.path.splitext(output_filename)
            if ext.lower() != ".mp3":
                output_filename = base + ".mp3"
                
            output_filepath = os.path.join(output_path, output_filename)

            print(f"  Attempting to extract audio using PyAV.")
            print(f"    Input file: {input_filepath}")
            print(f"    Output file: {output_filepath}")

            # Use PyAV to extract audio
            import av
            
            # Open the input file
            input_container = av.open(input_filepath)
            
            # Create the output container
            output_container = av.open(output_filepath, mode='w')
            
            # Add an audio stream to the output
            output_stream = output_container.add_stream('mp3')
            
            # Process the input audio
            for frame in input_container.decode(audio=0):
                # Encode the frame
                packet = output_stream.encode(frame)
                if packet:
                    output_container.mux(packet)
            
            # Flush any remaining packets
            packet = output_stream.encode(None)
            if packet:
                output_container.mux(packet)
            
            # Close the containers
            output_container.close()
            input_container.close()
            
            print(f"  Successfully extracted audio to {output_filepath}")
            return output_filepath  # Return the path to the extracted audio file
            
        except Exception as e:
            print(f"  An unexpected error occurred during audio extraction from {input_filepath}: {e}")
            return False

    def split_audio(self, audio_path: str, segments: list[dict]):
        """Splits the audio file into segments based on the start and end times."""
        try:
            print(f"Loading audio file for splitting: {audio_path}")
            print(f"Using device: {device.type}")
            y, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate
            print(f"Original sample rate: {sr} Hz")
            
            # Target sample rate for SNAC
            target_sr = 24000
            
            # Convert to tensor for processing
            waveform = torch.from_numpy(y).float()
            
            # Use torchaudio for resampling
            if sr != target_sr:
                print(f"Resampling from {sr} Hz to {target_sr} Hz using torchaudio")
                resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                sr = target_sr
            
            # Split the audio into segments
            chunks = []
            for segment in segments:
                # Convert time to samples
                start_time = segment["start"]
                end_time = segment["end"]
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                text = segment["text"]
                
                print(f"Processing segment: {start_time:.2f}s - {end_time:.2f}s")
                
                # Make sure we don't go out of bounds
                if start_sample >= len(waveform):
                    print(f"Warning: Start sample {start_sample} exceeds audio length {len(waveform)}")
                    continue
                    
                end_sample = min(end_sample, len(waveform))
                
                # Extract segment
                chunk = waveform[start_sample:end_sample]
                
                # Format tensor exactly as in the example: 
                # 1. First unsqueeze to make it [1, length]
                # 2. Then unsqueeze again to make it [1, 1, length]
                chunk_tensor = chunk.unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.inference_mode():
                    print(f"Encoding segment with SNAC, waveform shape: {chunk_tensor.shape}")
                    codes = snac_model.encode(chunk_tensor)
                    print(f"Generated codes with shape: {codes.shape if hasattr(codes, 'shape') else 'N/A'}")

                all_codes = []
                for i in range(codes[0].shape[1]):
                    all_codes.append(codes[0][0][i].item()+128266)
                    all_codes.append(codes[1][0][2*i].item()+128266+4096)
                    all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
                    all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
                    all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
                    all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
                    all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))
                
                chunks.append({"text": text.strip(), "all_codes": all_codes, "audio_duration": end_time - start_time})
            
            return chunks
            
        except Exception as e:
            print(f"Error in split_audio: {e}")
            import traceback
            traceback.print_exc()
            return []
        
if __name__ == "__main__":
    kb = ApiService()
    kb.authenticate(lambda: print("Autentificering gennemført"))
    # iterate over all pages of search results up
    # months = [1,2,3,4,5,6,7,8,9,10,11,12]
    # years = [2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]
    # for year in years:
    #     for month in months:
    #         total_results = kb.fetch_search_results(media_type="ds.tv", start_index=0, rows=10, year_start=year, year_end=year+1, month_number=month)["response"]["numFound"]
    #         print(f"Total results: {total_results}")
    #         
    #         total_pages = total_results // 100
    #         for page in range(1, total_pages):   
    #             print(f"Fetching page {page} of {total_pages}... {year} {month}")
    #             search_results = kb.fetch_search_results(media_type="ds.tv", start_index=page*100, rows=100, year_start=year, year_end=year+1, month_number=month)
    #             


    #             if search_results and isinstance(search_results, dict):
    #                 # Access the nested 'docs' list within 'response'
    #                 response_dict = search_results.get("response")
    #                 if response_dict and isinstance(response_dict, dict):
    #                     results_list = response_dict.get("docs")
    #                 else:
    #                     results_list = None
    #                     
    #                 if results_list is not None and isinstance(results_list, list):
    #                     print(f"Processing {len(results_list)} results...")
    #                     # list of entry_ids not in the database
    #                     ready_to_add = []
    #                     for result in results_list:
    #                         if isinstance(result, dict) and "kaltura_id" in result:
    #                             entry_id = result["kaltura_id"]
    #                             # Check if the entry_id is already in the database if not then insert it
    #                             if not collection.find_one({"kaltura_id": entry_id}):
    #                                 ready_to_add.append({"kaltura_id": entry_id, "year": year, "month": month})
    #                             else:
    #                                 print(f"Entry ID {entry_id} already exists in the database. Skipping...")
    #         
    #                     # batch adds
    #                     if len(ready_to_add) > 0:
    #                         collection.insert_many(ready_to_add)
    #                         print(f"Inserted {len(ready_to_add)} new entry IDs into the database.")
    #                     else:
    #                         print("No new entry IDs to insert.")
    #                   print(f"Fetching Kaltura data for entry ID: {entry_id}...")

    # Get all documents from the collection that does not have a "transcription" field
    documents = collection.find({"transcription": {"$exists": False}})

    for document in documents:
        print(document)
        entry_id = document["kaltura_id"]
        kaltura_data_str = kb.fetch_kaltura_data(entry_id)
        print(f"  Kaltura data: {kaltura_data_str}")
        
        if kaltura_data_str:
            # Extract the stream link using the existing method
            media_url = kb.extract_media_url_from_kaltura_response(kaltura_data_str)
            if media_url:
                print(f"  Stream link for {entry_id}: {media_url}")
                
                # Step 1: Download the MP4 file
                # Construct a filename for the MP4, e.g., kaltura_id.mp4
                # The file extension is already part of the media_url generation logic or defaults to mp4
                mp4_filename = f"{entry_id}.{media_url.split('.')[-1].split('?')[0] if '.' in media_url else 'mp4'}"
                downloaded_mp4_path = kb.download_media(media_url, mp4_filename, download_path="downloads")

                if downloaded_mp4_path:
                    # Step 2: Convert the downloaded MP4 to MP3
                    output_audio_filename = f"{entry_id}.mp3" # Output as mp3
                    extracted_audio_path = kb.extract_audio(downloaded_mp4_path, output_audio_filename, output_path="audio_files")

                    # Step 3: Transcribe the audio only if extraction was successful
                    if extracted_audio_path:
                        segments, info = batched_model.transcribe(extracted_audio_path, batch_size=16)
                        
                        print(f"Info: {info}")
                        segments_list = []
                        for segment in segments:
                            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                            segments_list.append({"start": segment.start, "end": segment.end, "text": segment.text})

                        # split the audio into the segments
                        chunks = kb.split_audio(extracted_audio_path, segments_list)

                        # save the chunks to the database
                        collection.update_one({"kaltura_id": entry_id}, {"$set": {"chunks": chunks}})

                        print(f"Transcription saved to the database for {entry_id}")

                        # Step 5: Delete the MP4 and MP3 files
                        os.remove(downloaded_mp4_path)
                        os.remove(extracted_audio_path)
                    else:
                        print(f"Skipping transcription for {entry_id} because audio extraction failed.")

                else:
                    print(f"  Skipping audio extraction for {entry_id} because MP4 download failed.")