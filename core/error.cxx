#include <Pantheon/error.h>

using namespace Pantheon;
using namespace Pantheon::DiagnosticService;

void Pantheon::DiagnosticService::PrintError(FILE *stream, enum ErrorType error)
{

}

ErrorType Pantheon::DiagnosticService::GetLastError()
{
    return ErrorType::Success;
}

void Pantheon::DiagnosticService::SetLastError(ErrorType error)
{

}