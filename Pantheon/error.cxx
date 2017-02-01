#include <Pantheon/error.h>

using namespace Pantheon;
using namespace Pantheon::DiagnosticService;

void Pantheon::DiagnosticService::PrintError(FILE *stream, enum PRESULT error)
{

}

PRESULT Pantheon::DiagnosticService::GetLastError()
{
    return PRESULT::OK;
}

void Pantheon::DiagnosticService::SetLastError(PRESULT error)
{

}