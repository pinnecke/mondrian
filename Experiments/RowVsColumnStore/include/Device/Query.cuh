#pragma once

extern "C"
{
    bool CopyDataToDevice(void **DestinationDevice, const void *SourceHost, size_t NumberOfBytes);
    bool CopyDataFromDevice(void **DestinationHost, const void *SourceDevice, size_t NumberOfBytes);
    bool FreeDataInDevice(void *DestinationDevice);
}


