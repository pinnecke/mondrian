#ifndef PANTHEON_ERROR_HPP
#define PANTHEON_ERROR_HPP

namespace pantheon
{
    enum class ErrorType
    {
        OK,
        UnresolvedConflict,
        UnsupportedOpertation,
        IllegalNullReference,
        ExplicitPrimiaryKeyMightNotHold,
        ForeignKeyConstraintViolated,
        PrimaryKeyConstraintViolated,
        ValueConstraintNotSatisfied
    };
}

#endif //PANTHEON_ERROR_HPP
